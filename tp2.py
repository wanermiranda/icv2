#!/bin/python2
import cv2
import numpy as np
import imagehelpers as ih
import threadingloops as tl
import sys
import getopt


MASK_SIZE = 500

MAX_HEIGHT = 300.00

IMG_COUNT = 6
IMG_PREFIX = 'img'
IMG_EXT = 'jpg'
# IMG_GROUND_TRUTH = 'groundtruth.jpg'


# The class used to collect the method used in the run, with its detectors and matcher
@ih.singleton
class Method:
    def __init__(self):
        self._method = -1
        self._detector = None
        self._matcher = None
        self._extractor = None

    def set(self, method=ih.SIFT_SIFT):
        self._method = method
        print method

        if method == ih.FAST_BRIEF:
            self._detector = cv2.FastFeatureDetector()
            self._extractor = cv2.DescriptorExtractor_create("BRIEF")
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if method == ih.SIFT_SIFT:
            self._detector = cv2.SIFT(400)
            # Faster than brute force, considering that sift is a very expensive detector
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=64)
            # cv2.FlannBasedMatcher(index_params, search_params)
            self._matcher = cv2.BFMatcher()

        if method == ih.ORB_ORB:
            self._detector = cv2.ORB(edgeThreshold=60)
            # HAMMING because orb create a set binary descriptors
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_extractor(self):
        return self._extractor

    def get_detector(self):
        return self._detector

    def get_matcher(self):
        return self._matcher

    def get_method(self):
        return self._method

    def get_good_matches(self, matches, threshold):
        good = []
        if self._method == ih.SIFT_SIFT:
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good.append(m)
        else:
            print 'Sorting'
            matches = sorted(matches, key=lambda x: x.distance)
            # for match in matches:
            # print match.distance
            good = matches[:10]
            print 'finished sort'

        return good


class ImageBlock:
    def __init__(self, path=None, image=None):
        self._descriptors = None
        self._keypoints = None
        self._path = None
        self._homography = None

        if image is not None:
            self._path = path
            self._image = image
        else:
            self._path = path
            self._image = cv2.imread(self._path)
            h, w = self._image.shape[:2]
            if h > MAX_HEIGHT:
                scale = MAX_HEIGHT / h
                self._image = cv2.resize(self._image, None, fx=scale, fy=scale)

    def get_image(self):
        return self._image

    def get_descriptors(self):
        return self._descriptors

    def get_keypoints(self):
        return self._keypoints

    def get_path(self):
        return self._path

    def detect(self, left=None, right=None):

        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]

        mask = np.zeros((img_h, img_w), np.uint8)

        if (left is None) and (right is None):
            mask.fill(255)

        if left is not None:
            mask[:left, :] = 255

        if right is not None:
            mask[:img_w - right, :] = 255

        # detecting key points and regions
        detector = Method().get_detector()

        print "detect " + self.get_path()
        kp = detector.detect(gray)  # , mask)
        print "compute " + self.get_path()
        if Method().get_method() == ih.FAST_BRIEF:
            (kp, desc) = Method().get_extractor().compute(gray, kp)  # , mask)
        else:
            (kp, desc) = detector.compute(gray, kp)  # , mask)
        self._keypoints = kp
        self._descriptors = desc

    def match(self, target, threshold=0.7):
        print "Matching"
        matcher = Method().get_matcher()
        if Method().get_method() == ih.SIFT_SIFT:
            matches = matcher.knnMatch(self._descriptors, target.get_descriptors(), k=2)
        else:
            matches = matcher.match(self._descriptors, target.get_descriptors())

        good = Method().get_good_matches(matches, threshold)
        return good

    def get_homography(self):
        return self._homography

    def set_homography(self, homography):
        self._homography = homography

    def gen_homography(self, target, good_matches):
        # filter only the good matches and reshape the vector for the findHomography
        src_pts = np.float32([(self._keypoints[m.queryIdx]).pt for m in good_matches]).reshape(-1, 1, 2)

        target_kp = target.get_keypoints()
        # filter only the good matches and reshape the vector for the findHomography
        dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        print "homography"
        matrix_h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        print(ih.determinant(matrix_h))

        return matrix_h

    def get_new_dimensions(self, h):
        return ih.get_dimensions(h, self.get_image())

    def blend(self, target_block, name, blend_type=ih.AVERAGE):
        result_image = self.internal_blend(target=target_block, blend_type=blend_type)
        result_block = ImageBlock(name, image=result_image)
        ih.save_image(result_image, name)
        return result_block

    def internal_blend(self, target, blend_type=ih.AVERAGE):
        print "Matching " + self.get_path() + " and " + target.get_path()
        loc_h, loc_w = self.get_image().shape[:2]
        matches = target.match(self)
        # ih.save_image(self.get_image(), 'self.png')
        # ih.save_image(target.get_image(), 'target.png')

        self._homography = target.gen_homography(self, matches)

        min_y, min_x, new_h, new_w = target.get_new_dimensions(self._homography)
        print min_y, min_x, new_h, new_w

        if new_h < loc_h:
            new_h = loc_h + min_y

        if new_w < loc_w:
            new_w = loc_w + min_x

        h_inv = np.linalg.inv(self._homography)

        identity = np.identity(3)

        arguments = h_inv, min_x, min_y, identity, self.get_image(), blend_type

        enlarged_base_img = np.zeros((new_h, new_w, 3), np.uint8)

        print 'warping'
        tl.ThreadingLoopsImage(4, enlarged_base_img, target.get_image(), ih.warp_and_blend, arguments).execute()

        return enlarged_base_img


class Mosaic:
    def __init__(self, method, blend_type):
        Method().set(method)
        block_list = []
        for img_idx in range(1, IMG_COUNT + 1):
            img_path = IMG_PREFIX + str(img_idx) + "." + IMG_EXT
            print img_path
            img_block = ImageBlock(path=img_path)
            img_block.detect()
            block_list.append(img_block)

        result = block_list[2].blend(block_list[3], str(method) + '_' + str(blend_type) + '_step1.png', blend_type)
        result.detect(left=MASK_SIZE)

        result = result.blend(block_list[1], str(method) + '_' + str(blend_type) + '_step2.png', blend_type)
        result.detect(left=MASK_SIZE)

        result = result.blend(block_list[0], str(method) + '_' + str(blend_type) + '_step3.png', blend_type)
        result.detect(right=MASK_SIZE)

        result = result.blend(block_list[4], str(method) + '_' + str(blend_type) + '_step4.png', blend_type)
        result.detect(right=MASK_SIZE)

        result.blend(block_list[5], str(method) + '_' + str(blend_type) + '_step5.png', blend_type)

if __name__ == "__main__":
    arg_list = sys.argv[1:]
    mtd = ih.SIFT_SIFT
    bld = ih.MERGE
    try:
        opts, args = getopt.getopt(arg_list, "hm:b:", ["method=", "blending="])
    except getopt.GetoptError:
        print 'example: tp2.py -m 0 -b 0 ' \
              '\n --method: 0 = SIFT_SIFT, 1 = FAST_BRIEF, 2 = ORB_ORB ' \
              '\n --blender: 0 = MERGE, 1 = AVERAGE, 2 = FEATHERING'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'tp2.py -m 0 -b 0 ' \
                  '\n --method: 0 = SIFT_SIFT, 1 = FAST_BRIEF, 2 = ORB_ORB ' \
                  '\n --blender: 0 = MERGE, 1 = AVERAGE, 2 = FEATHERING'
            sys.exit()
        elif opt in ("-m", "--method"):
            mtd = int(arg)
        elif opt in ("-b", "--blending"):
            bld = int(arg)
    Mosaic(mtd, bld)
