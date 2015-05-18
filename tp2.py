# !/bin/python
import cv2
import numpy as np
import math
import imagehelpers as ih
import threadingloops as tl

MASK_SIZE = 3000

MAX_HEIGHT = 1600.00

SIFT_SIFT = 0
FAST_BRIEF = 1
ORB_ORB = 2

CURRENT_METHOD = 0

IMG_COUNT = 6
IMG_PREFIX = 'img'
IMG_EXT = 'jpg'
IMG_GROUND_TRUTH = 'groundtruth.jpg'
FLANN_INDEX_KDTREE = 1


# The class used to collect the method used in the run, with its detectors and matcher
@ih.singleton
class Method:
    def __init__(self):
        self._method = -1
        self._detector = None
        self._matcher = None
        self._extractor = None

    def set(self, method=SIFT_SIFT):
        self._method = method
        print method

        if method == FAST_BRIEF:
            self._detector = cv2.FastFeatureDetector()
            self._extractor = cv2.DescriptorExtractor_create("BRIEF")
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if method == SIFT_SIFT:
            self._detector = cv2.SIFT(400)
            # Faster than brute force, considering that sift is a very expensive detector
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=64)
            # cv2.FlannBasedMatcher(index_params, search_params)
            self._matcher = cv2.BFMatcher()

        if method == ORB_ORB:
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
        if self._method == SIFT_SIFT:
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
        if Method().get_method() == FAST_BRIEF:
            (kp, desc) = Method().get_extractor().compute(gray, kp)  # , mask)
        else:
            (kp, desc) = detector.compute(gray, kp)  # , mask)
        self._keypoints = kp
        self._descriptors = desc

    def match(self, target, threshold=0.7):
        print "Matching"
        matcher = Method().get_matcher()
        if Method().get_method() == SIFT_SIFT:
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

        img_h, img_w = self.get_image().shape[:2]

        base_p1 = [0, 0]
        base_p2 = [img_w, 0]
        base_p3 = [0, img_h]
        base_p4 = [img_w, img_h]

        max_x = 0
        max_y = 0
        min_x = 0
        min_y = 0

        for pt in [base_p1, base_p2, base_p3, base_p4]:

            # transposed matrix
            normal_pt = ih.warp_pure_pt(h, pt)

            if normal_pt[0] > max_x:
                max_x = normal_pt[0]

            if normal_pt[1] > max_y:
                max_y = normal_pt[1]

            if normal_pt[0] < min_x:
                min_x = normal_pt[0]

            if normal_pt[1] < min_y:
                min_y = normal_pt[1]


        min_x *= -1
        min_y *= -1

        max_x += min_x
        max_y += min_y

        print min_y, min_x, max_y, max_x
        return min_y, min_x, max_y, max_x

    def warp_pure(self, h, new_h, new_w, min_u=0, min_v=0):
        print 'Warp'
        h_inv = np.linalg.inv(h)
        warped_img = np.zeros((new_h, new_w, 3), np.uint8)

        args = h_inv, min_u, min_v, new_h, new_w
        tl.ThreadingLoopsImage(4, warped_img, self.get_image(), ih.warp_pure_pixel, args).execute()
        return warped_img

    def combine(self, target_block, name):
        result_image = self.get_wrap_image(target=target_block)
        result_block = ImageBlock(name, image=result_image)
        ih.save_image(result_image, name)
        return result_block

    def warp_target(self, homography, target):
        loc_h, loc_w = self.get_image().shape[:2]
        min_y, min_x, new_h, new_w = target.get_new_dimensions(homography)
        min_y = int(math.ceil(min_y))
        min_x = int(math.ceil(min_x))
        new_w = int(math.ceil(new_w))
        new_h = int(math.ceil(new_h))
        if new_h < loc_h:
            new_h = loc_h + min_y
        if new_w < loc_w:
            new_w = loc_w + min_x
        target_img_wrp = target.warp_pure(homography, new_h, new_w, min_x, min_y)
        ih.save_image(target_img_wrp, 'wrapped_' + target.get_path())
        return min_x, min_y, new_h, new_w, target_img_wrp

    def blend_image(self, target, homography):

        min_x, min_y, new_h, new_w, target_img_wrp = self.warp_target(homography, target)

        ih.save_image(target_img_wrp, 'wrapped_' + target.get_path())

        self_img_wrp = self.warp_pure(np.identity(3), new_h, new_w, min_x, min_y)
        ih.save_image(self_img_wrp, 'wrapped_' + self.get_path())

        (ret, data_map) = cv2.threshold(cv2.cvtColor(self_img_wrp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = np.zeros((new_h, new_w, 3), np.uint8)

        enlarged_base_img = cv2.add(enlarged_base_img, target_img_wrp,
                                    mask=np.bitwise_not(data_map),
                                    dtype=cv2.CV_8U)

        ih.save_image(enlarged_base_img, 'wrapped_1_' + self.get_path())

        final_img = cv2.add(enlarged_base_img, self_img_wrp,
                            dtype=cv2.CV_8U)

        ih.save_image(final_img, 'wrapped_2_' + self.get_path())

        return final_img

    def get_wrap_image(self, target):
        print "Matching " + self.get_path() + " and " + target.get_path()
        loc_h, loc_w = self.get_image().shape[:2]
        matches = target.match(self)
        ih.save_image(self.get_image(), 'self.jpg')
        ih.save_image(target.get_image(), 'target.jpg')

        self._homography = target.gen_homography(self, matches)

        min_y, min_x, new_h, new_w = target.get_new_dimensions(self._homography)

        min_y = int(math.ceil(min_y))
        min_x = int(math.ceil(min_x))

        new_w = int(math.ceil(new_w))
        new_h = int(math.ceil(new_h))

        if new_h < loc_h:
            new_h = loc_h + min_y

        if new_w < loc_w:
            new_w = loc_w + min_x

        # target_img_wrp = cv2.warpPerspective(target.get_image(), self._homography, (new_w, new_h))
        # target.warp(self._homography, new_h, new_w)
        target_img_wrp = target.warp_pure(self._homography, new_h, new_w, min_x, min_y)
        # target_img_wrp = ih.apply_median(target_img_wrp, False)
        ih.save_image(target_img_wrp, 'wrapped_' + target.get_path())

        self_img_wrp = self.warp_pure(np.identity(3), new_h, new_w, min_x, min_y)
        ih.save_image(self_img_wrp, 'wrapped_' + self.get_path())

        (ret, data_map) = cv2.threshold(cv2.cvtColor(self_img_wrp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = np.zeros((new_h, new_w, 3), np.uint8)

        enlarged_base_img = cv2.add(enlarged_base_img, target_img_wrp,
                                    mask=np.bitwise_not(data_map),
                                    dtype=cv2.CV_8U)

        ih.save_image(enlarged_base_img, 'no_median_wrapped_1_' + self.get_path())

        # ih.apply_median(enlarged_base_img, False)

        ih.save_image(enlarged_base_img, 'wrapped_1_' + self.get_path())

        enlarged_base_img = cv2.fastNlMeansDenoisingColored(enlarged_base_img)

        final_img = cv2.add(enlarged_base_img, self_img_wrp,
                            dtype=cv2.CV_8U)

        ih.save_image(final_img, 'wrapped_2_' + self.get_path())

        return final_img


class Mosaic:

    def __init__(self):
        Method().set(CURRENT_METHOD)
        block_list = []
        for img_idx in range(1, IMG_COUNT + 1):
            img_path = IMG_PREFIX + str(img_idx) + "." + IMG_EXT
            print img_path
            img_block = ImageBlock(path=img_path)
            img_block.detect()
            block_list.append(img_block)

        h_list = []

        for idx in range(0, IMG_COUNT-1):
            next_img = block_list[idx+1]
            main_img = block_list[idx]
            matches = next_img.match(main_img)
            homography = next_img.gen_homography(main_img, matches)
            h_list.append(homography)

        # result = ImageBlock('img12.jpg', image=block_list[0].blend_image(block_list[1], h_list[0]))
        #
        # min_x, min_y, new_h, new_w, target_img_wrp = block_list[2].warp_target(h_list[0], block_list[2])
        #
        # result = ImageBlock('img2_warp.jpg', target_img_wrp)
        # result = ImageBlock('img123.jpg',  block_list[2].blend_image(result, h_list[1]))

        result = block_list[2].combine(block_list[3], 'step1')
        result.detect(left=MASK_SIZE)

        # #
        result = result.combine(block_list[1], 'step2')
        result.detect(left=MASK_SIZE)
        #
        result = result.combine(block_list[0], 'step3')
        result.detect(right=MASK_SIZE)

        result = result.combine(block_list[4], 'step4')
        result.detect(right=MASK_SIZE)

        result = result.combine(block_list[5], 'step5')
        result.detect(right=MASK_SIZE)

        # for b, t in img_combinations:
        #
        # block12 = block_list[0].combine(target_block=block_list[1], name='img12.png')
        # block12.detect(right=MASK_SIZE)
        #
        # block123 = block12.combine(target_block=block_list[2], name='img123.png')
        # block123.detect(right=MASK_SIZE)
        #
        # block1234 = block123.combine(target_block=block_list[3], name='img1234.png')
        # block1234.detect(right=MASK_SIZE)
        #
        # block12345 = block1234.combine(target_block=block_list[4], name='img12345.png')
        # block12345.detect(right=MASK_SIZE)
        #
        # block123456 = block1234.combine(target_block=block_list[5], name='img123456.png')
        # block123456.detect(right=MASK_SIZE)


def get_dimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if (max_x is None) or (normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if (max_y is None) or (normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if (min_x is None) or (normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if (min_y is None) or (normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return min_x, min_y, max_x, max_y


if __name__ == "__main__":
    Mosaic()

__author__ = 'gorigan'


# junk yard

# def print_lines(self, target, homography, mask, good_matches):
#
# matches_mask = mask.ravel().tolist()
#
# h, w = self._image.shape[:2]
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, homography)
#
#     target_img = cv2.polylines(target.get_image(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
#
#     # draw_params = dict(matchColor=(0, 255, 0),
#     #                    singlePointColor = None,
#     #                    matchesMask = matches_mask,
#     #                    flags = 2)
#
#     result = cv2.drawMatches(  #self._image, self._keypoints,
#                                self._image, None,
#                                #target_img, target.get_keypoints(),
#                                target_img, None,
#                                None, None, None)
#     #good_matches, None, **draw_params)
#
#     show_image(result)
