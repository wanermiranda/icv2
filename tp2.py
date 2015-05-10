# !/bin/python
import cv2
import numpy as np
import math
import os
from itertools import combinations
import glob
import threading
import time
import matplotlib.pyplot as plt


IMG_COUNT = 6
IMG_PREFIX = 'img'
IMG_EXT = 'jpg'
IMG_GROUND_TRUTH = 'groundtruth.jpg'
FLANN_INDEX_KDTREE = 1

# Funcao de suporte para salvar a imagem
def save_image(img, name):
    plt.imshow(img)
    plt.savefig(name)


# funcao de suporte para exibicao de imagem
def show_image(img):
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.show()


# Funcao de suporte para exibicao da imagem
# funcao que faz o calculo do histograma de cada canal de cor
# utilizado somente a funcao do opencv por ser mais rapida,
# mas o algoritmo segue abaixo como demonstracao
def image_hist(img):
    colors_hist = np.zeros((3, 256))
    for i in range(0, 3):
        colors_hist[i, :256] = cv2.calcHist([img], [i], None, [256], [0, 256])[:256, 0]
        max_bin = colors_hist[i, :256].max()
        colors_hist[i, :256] /= max_bin
    return colors_hist


# funcao de suporte para o calculo do erro quadratico medio
def get_mse(query, target):
    (m, n) = query.shape[:2]
    if (m == 3) and (n == 256):
        query = query[0:3, 0:256]
        target = target[0:3, 0:256]
    # (m, n) = query.shape[:2]
    # print str(m) + ", " + str(n)
    sums = np.power(np.array(query) - np.array(target), 2).sum()
    return sums / (m * n)


# funcao de suporte para calcular a relacao entre uma
# imagem alvo, uma imagem de pesquisa para decidir o tamanho da janela de percorrer
def get_size_factor(query, target, factor):
    (target_h, target_w) = target.shape[:2]
    (query_h, query_w) = query.shape[:2]
    if query_h > query_w:
        factor = (factor * target_h) / query_h
    else:
        factor = (factor * target_w) / query_w
    return factor


# funcao de suporte para recuperar qual a relacao entre o tamanho da imagem alvo
# e o tamanho desejado para redimensionamento
def get_base_scale(img, maximum):
    (query_h, query_w) = img.shape[:2]
    if query_h > query_w:
        factor = maximum / query_h
    else:
        factor = maximum / query_w
    return factor


# filtro de passa baixa
def remove_noise(img):
    kernel = np.ones((3, 3), np.float32) / 8
    dst = cv2.filter2D(img, -1, kernel)
    return dst


class ImageBlock:
    def __init__(self, path=None, image=None):
        self._descriptors = None
        self._keypoints = None
        self._path = None

        if image is not None:
            self._path = path
            self._image = image
        else:
            self._path = path
            self._image = cv2.imread(self._path)

        h, w = self._image.shape[:2]

        if h > 800:
            scale = 800.00 / h
            self._image = cv2.resize(self._image, None, fx=scale, fy=scale)


    def get_image(self):
        return self._image

    def get_descriptors(self):
        return self._descriptors

    def get_keypoints(self):
        return self._keypoints

    def get_path(self):
        return self._path

    @staticmethod
    def get_detector():
        return cv2.xfeatures2d.SIFT_create()

    @staticmethod
    def get_matcher():
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=32)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher

    def detect(self, left=None, right=None):

        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]

        mask = np.zeros((img_h, img_w), np.uint8)

        if (left is None) and (right is None):
            mask.fill(255)

        if left is not None:
            mask[:, :left] = 255

        if right is not None:
            mask[img_h-right:, :] = 255

        # detecting key points and regions
        detector = self.get_detector()

        print "detect " + self.get_path()
        kp = detector.detect(gray, mask)
        print "compute " + self.get_path()
        (kp, desc) = detector.compute(gray, kp, mask)
        self._keypoints = kp
        self._descriptors = desc

    @staticmethod
    def determinant(homography):
        return (homography[0, 1] * homography[1, 1] * homography[2, 2]) - (
            homography[2, 0] * homography[1, 1] * homography[0, 2])

    def match(self, target, threshold=0.8):
        print "Matching"
        matcher = self.get_matcher()
        knn_matches = matcher.knnMatch(self._descriptors, target.get_descriptors(), k=2)

        good = []
        for m, n in knn_matches:
            if m.distance < threshold * n.distance:
                good.append(m)
        return good

    def get_homography(self, target, good_matches):
        src_pts = np.float32([(self._keypoints[m.queryIdx]).pt for m in good_matches]).reshape(-1, 1, 2)
        target_kp = target.get_keypoints()
        dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        print "homography"
        matrix_h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)

        print(self.determinant(matrix_h))

        # self.print_lines(target, matrix_h, mask, good_matches)

        return matrix_h



class Mosaic:
    def get_wrap_image(self, center_block, target):
        print "Matching " + center_block.get_path() + " and " + target.get_path()
        matches = center_block.match(target)
        homography = center_block.get_homography(target, matches)
        inv_homography = np.linalg.inv(homography)
        (min_x, min_y, max_x, max_y) = findDimensions(target.get_image(), inv_homography)
        # Adjust max_x and max_y by base img size
        max_x = max(max_x, center_block.get_image().shape[1])
        max_y = max(max_y, center_block.get_image().shape[0])
        move_h = np.matrix(np.identity(3), np.float32)
        if min_x < 0:
            move_h[0, 2] += -min_x
            max_x += -min_x
        if min_y < 0:
            move_h[1, 2] += -min_y
            max_y += -min_y

        mod_inv_h = move_h * inv_homography

        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))
        target_img_wrp = cv2.warpPerspective(target.get_image(), mod_inv_h, (img_w, img_h))

        (ret, data_map) = cv2.threshold(cv2.cvtColor(target_img_wrp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)
        # img = cv2.half()
        # cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
        # show_image(target_img_wrp)

        center_img_wrp = cv2.warpPerspective(center_block.get_image(), move_h, (img_w, img_h))

        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
        # Now add the warped image
        save_image(target_img_wrp, 'wrapped_' + target.get_path())
        save_image(center_img_wrp, 'wrapped_' + center_block.get_path())

        enlarged_base_img = cv2.add(enlarged_base_img, target_img_wrp,
                                    dtype=cv2.CV_8U)
        save_image(enlarged_base_img, 'wrapped_1_' + center_block.get_path())

        final_img = cv2.add(enlarged_base_img, center_img_wrp,
                            dtype=cv2.CV_8U)

        save_image(final_img, 'wrapped_2_' + center_block.get_path())
        # img = cv2.half()
        # cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
        return final_img

    def combine(self, center_block, target_block, name):
        result_image = self.get_wrap_image(center_block, target_block)
        result_block = ImageBlock(name, image=result_image)
        save_image(result_image, name)
        return result_block

    def __init__(self):

        center = (IMG_COUNT + 1) / 2

        block_list = []
        for img_idx in range(1, IMG_COUNT + 1):
            img_path = IMG_PREFIX + str(img_idx) + "." + IMG_EXT
            print img_path
            img_block = ImageBlock(path=img_path)
            img_block.detect()
            block_list.append(img_block)

        # for b, t in img_combinations:

        block34 = self.combine(block_list[2], block_list[3], 'img34.png')
        block34.detect(left=3000)
        block234 = self.combine(block34, block_list[1], 'img234.png')
        del block34
        block234.detect(left=3000)
        block1234 = self.combine(block234, block_list[0], 'img1234.png')
        del block234
        block1234.detect(right=3000)
        block12345 = self.combine(block1234, block_list[4], 'img12345.png')
        del block1234
        block12345.detect(right=3000)
        block123456 = self.combine(block12345, block_list[5], 'img123456.png')
        del block12345
        # block123456.detect()
        show_image(block123456.get_image())
        del block123456
        # left = block_list[2]
        # center_block = block_list[3]
        # right = block_list[4]
        #
        # center_img_1 = self.left_right(center_block, left, right)
        # show_image(center_img_1)
        # left = block_list[1]
        # right = block_list[5]
        # center_block_1 = ImageBlock('center_image_1', center_img_1)
        #
        # center_img_2 = self.left_right(center_block_1, left, right)

        #show_image(center_img_2)


def findDimensions(image, homography):
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

        if ( max_x == None or normal_pt[0, 0] > max_x ):
            max_x = normal_pt[0, 0]

        if ( max_y == None or normal_pt[1, 0] > max_y ):
            max_y = normal_pt[1, 0]

        if ( min_x == None or normal_pt[0, 0] < min_x ):
            min_x = normal_pt[0, 0]

        if ( min_y == None or normal_pt[1, 0] < min_y ):
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)


if __name__ == "__main__":
    Mosaic()

__author__ = 'gorigan'



#junk yard

        # def print_lines(self, target, homography, mask, good_matches):
        #
        # matches_mask = mask.ravel().tolist()
        #
        #     h, w = self._image.shape[:2]
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
