import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

SIFT_SIFT = 0
FAST_BRIEF = 1
ORB_ORB = 2

MERGE = 0
AVERAGE = 1
FEATHERING = 2


# Support function to copy an image to another new with new size
def copy_with(img, h, w):
    base_image = np.zeros((h, w, 3), np.uint8)
    img_h, img_w = img.shape[:2]
    base_image[:img_h, :img_w] = img[:img_h, :img_w]
    return base_image


# support function to load image till the max height
def load_image(path, max_height=None):
    result_image = cv2.imread(path)
    h, w = result_image.shape[:2]

    if not (max_height is None):
        if h > max_height:
            scale = max_height / h
            result_image = cv2.resize(result_image, None, fx=scale, fy=scale)

    return result_image


# support function calc the homography determinant value
def determinant(homography):
    return (homography[0, 1] * homography[1, 1] * homography[2, 2]) - (
        homography[2, 0] * homography[1, 1] * homography[0, 2])


# Support function to save images to files
def save_image(img, name):
    cv2.imwrite(name, img)


# Support function to show images
def show_image(img):
    plt.imshow(img)
    plt.show()


# Support function to calc color histograms, if needed
def image_hist(img):
    colors_hist = np.zeros((3, 256))
    for i in range(0, 3):
        colors_hist[i, :256] = cv2.calcHist([img], [i], None, [256], [0, 256])[:256, 0]
        max_bin = colors_hist[i, :256].max()
        colors_hist[i, :256] /= max_bin
    return colors_hist


# Support function to calculate the difference image
def get_diff_image(ground_truth, target):
    return cv2.subtract(ground_truth, target)

# Support function to calculate the mean squared error
def get_mse(ground_truth, target):
    (m, n) = ground_truth.shape[:2]
    if (m == 3) and (n == 256):
        ground_truth = ground_truth[0:3, 0:256]
        target = target[0:3, 0:256]
    # (m, n) = query.shape[:2]
    # print str(m) + ", " + str(n)
    sums = np.power(np.array(ground_truth) - np.array(target), 2).sum()
    return sums / (m * n)


# Support function to define the correlation factor between to images
def get_size_factor(query, target, factor):
    (target_h, target_w) = target.shape[:2]
    (query_h, query_w) = query.shape[:2]
    if query_h > query_w:
        factor = (factor * target_h) / query_h
    else:
        factor = (factor * target_w) / query_w
    return factor


# Support function to define if the image surpass a limited size
def get_base_scale(img, maximum):
    (query_h, query_w) = img.shape[:2]
    if query_h > query_w:
        factor = maximum / query_h
    else:
        factor = maximum / query_w
    return factor


# Low pass filter
def remove_noise(img):
    kernel = np.ones((3, 3), np.float32) / 8
    dst = cv2.filter2D(img, -1, kernel)
    return dst


# To keep the code clear applied to design patterns, decorator and singleton
# since the method is selected once in a run
def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance


# A function that checks if the point is inside the image
# Input params:
# x = position for the loop in the image
# y = position for the loop in the image
# img = used to get the inbox size
# Output Params
# true if the x and y are inside the image
def in_bounds(x, y, img):
    img_h, img_w = img.shape[:2]
    return (x < img_w) and (y < img_h) and (x >= 0) and (y >= 0)


# A function that warp the corners to get the max and values for the new image
# Input params:
# h = Homography
# x = position for the loop in the big image
# y = position for the loop in the big image
#
# Output Params
# u, v = coordinates on the original image
def get_dimensions(h, img):
        img_h, img_w = img.shape[:2]

        corner_tl = [0, 0]
        corner_tr = [img_w, 0]
        corner_bl = [0, img_h]
        corner_br = [img_w, img_h]

        max_x = 0
        max_y = 0
        min_x = 0
        min_y = 0

        for pt in [corner_tl, corner_tr, corner_bl, corner_br]:

            # transposed matrix
            normal_pt = warp_pure_pt(h, pt)

            if normal_pt[0] > max_x:
                max_x = normal_pt[0]

            if normal_pt[1] > max_y:
                max_y = normal_pt[1]

            if normal_pt[0] < min_x:
                min_x = normal_pt[0]

            if normal_pt[1] < min_y:
                min_y = normal_pt[1]

        min_x = math.ceil(min_x * -1)
        min_y = math.ceil(min_y * -1)

        max_x = math.ceil(max_x + min_x)
        max_y = math.ceil(max_y + min_y)

        return min_y, min_x, max_y, max_x


# A function that is used to calc the max, min and difference between values
# Input params:
# value1 = a number or coordinate in the same axis of the other value
# value2 = a number or coordinate in the same axis of the other value
# Output Params
# returns the minimum value and maximum value considering the difference between them
def intersection_coord(value1, value2):

    if value1 > value2:
        max_value = value1
        min_value = value2
    else:
        max_value = value2
        min_value = value1

    diff = max_value - min_value
    return (min_value + diff), (max_value - diff)


# A function that is used to calc the ROI for the blending
# Input params:
# img1_new_dim = new dimensions for the image 1 (min and max values in a tuple)
# img2_new_dim = new dimensions for the image 2 (min and max values in a tuple)
# Output Params
# Return two sets of min and max values for x and y defining the roi
def get_roi(img1_new_dim, img2_new_dim):
    min1_y, min1_x, max1_y, max1_x = img1_new_dim
    min2_y, min2_x, max2_y, max2_x = img2_new_dim

    if most_left(img1_new_dim, img2_new_dim):
        roi_x = intersection_coord(max1_x, min2_x)
    else:
        roi_x = intersection_coord(max2_x, min1_x)

    if most_bottom(img1_new_dim, img2_new_dim):
        roi_y = intersection_coord(max1_y, min2_y)
    else:
        roi_y = intersection_coord(max2_y, min1_y)

    return roi_x, roi_y


# A function that is used to check which image is on left
# Input params:
# img1_new_dim = new dimensions for the image 1 (min and max values in a tuple)
# img2_new_dim = new dimensions for the image 2 (min and max values in a tuple)
# Output Params
# True if the im1 is on left, false otherwise
def most_left(img1_new_dim, img2_new_dim):
    min1_y, min1_x, max1_y, max1_x = img1_new_dim
    min2_y, min2_x, max2_y, max2_x = img2_new_dim
    return max1_x < max2_x


# A function that is used to check which image is on bottom
# Input params:
# img1_new_dim = new dimensions for the image 1 (min and max values in a tuple)
# img2_new_dim = new dimensions for the image 2 (min and max values in a tuple)
# Output Params
# True if the im1 is on bottom, false otherwise
def most_bottom(img1_new_dim, img2_new_dim):
    min1_y, min1_x, max1_y, max1_x = img1_new_dim
    min2_y, min2_x, max2_y, max2_x = img2_new_dim
    return max1_y < max2_y


# A function that is used to calc weight for the feathering blend
# The weight is calculated based on
# L1(P1 - BORDER1) + L1(P2 - BORDER2) * SHARP
# L1 = L1 distance or Manhattan distance
# Input params:
# x, y = the point in the big image
# img1_new_dim = new dimensions for the image 1 (min and max values in a tuple)
# img2_new_dim = new dimensions for the image 2 (min and max values in a tuple)
# roi = the region of interest that will be feathered and used to calc the weights
# sharp = constant value to reduce the impact during the feathering
# Output Params
# the weight value to be applied during the feathearing
def get_weight(img1_new_dim, img2_new_dim, roi, x, y, sharp=0.02):
    (roi_x1, roi_x2), (roi_y1, roi_y2) = roi
    if (roi_x1 <= x <= roi_x2) and (roi_y1 <= y <= roi_y2):
        if most_left(img1_new_dim, img2_new_dim):
            w1 = roi_x2 - x
        else:
            w1 = roi_x1 + x

        if most_bottom(img1_new_dim, img2_new_dim):
            w1 += roi_y2 - y
        else:
            w1 += roi_y1 + y

        w1 *= sharp
    else:
        w1 = 1

    # threshold
    if w1 > 1:
        w1 = 1
    return w1


# A function that is used to calc the blend merging points
# for the target and the main image
# P = P1(x,y) OR P2(x,y)
# Input params:
# img1_pt = the point in the target img1
# img1 = target image
# img2_pt = the point in the target img2
# img2 = target image
#
# Output Params
# The new pixel value (b, g, r)
def get_one_of(target_pt, img1, main_pt, img2):
        x1, y1 = target_pt
        x2, y2 = main_pt

        b = 0
        g = 0
        r = 0

        if in_bounds(x1, y1, img1):
            b = img1[y1, x1][0]
            g = img1[y1, x1][1]
            r = img1[y1, x1][2]
        else:
            if in_bounds(x2, y2, img2):
                b = img2[y2, x2][0]
                g = img2[y2, x2][1]
                r = img2[y2, x2][2]

        return np.array([b, g, r])


# A function that is used to calc the blend averaging points
# for the target and the main image
# P = (P1(x,y) + P2(x,y) / 2
# Input params:
# img1_pt = the point in the target img1
# img1 = target image
# img2_pt = the point in the target img2
# img2 = target image
#
# Output Params
# The new pixel value (b, g, r)
def get_avg_2point(img1_pt, img1, img2_pt, img2):
        x1, y1 = img1_pt
        x2, y2 = img2_pt

        div = 2

        b1 = 0
        g1 = 0
        r1 = 0

        b2 = 0
        g2 = 0
        r2 = 0

        if in_bounds(x1, y1, img1):
            b1 = img1[y1, x1][0]
            g1 = img1[y1, x1][1]
            r1 = img1[y1, x1][2]
        else:
            div = 1

        if in_bounds(x2, y2, img2):
            b2 = img2[y2, x2][0]
            g2 = img2[y2, x2][1]
            r2 = img2[y2, x2][2]
        else:
            div = 1

        if ((b1 + g1 + r1) == 0) or ((b2 + g2 + r2) == 0):
            div = 1

        b = (b1 + b2) / div
        g = (g1 + g2) / div
        r = (r1 + r2) / div

        return np.array([b, g, r])


# A function that is used to calc the blend feathering point
# for the target and the main image
#
# Input params:
# img1_pt = the point in the target img1
# img1 = target image
# img1_new_dim = new dimensions for the image 1 (min and max values in a tuple)
# img2_pt = the point in the target img2
# img2 = target image
# img2_new_dim = new dimensions for the image 2 (min and max values in a tuple)
# roi = the region of interest that will be feathered and used to calc the weights
#
# Output Params
# The new pixel value (b, g, r)
def feathering_2point(img1_pt, img1, img1_new_dim, img2_pt, img2, img2_new_dim, roi):

        x1, y1 = img1_pt
        x2, y2 = img2_pt

        b1 = 0
        g1 = 0
        r1 = 0

        b2 = 0
        g2 = 0
        r2 = 0

        if in_bounds(x1, y1, img1):
            b1 = img1[y1, x1][0]
            g1 = img1[y1, x1][1]
            r1 = img1[y1, x1][2]
            w1 = get_weight(img1_new_dim, img2_new_dim, roi, x1, y1)
            div = w1
        else:
            div = 1
            w1 = 0

        if in_bounds(x2, y2, img2):
            b2 = img2[y2, x2][0]
            g2 = img2[y2, x2][1]
            r2 = img2[y2, x2][2]
            w2 = get_weight(img1_new_dim, img2_new_dim, roi, x2, y2)
            div += w2
        else:
            div = 1
            w2 = 0

        if ((b1 + g1 + r1) == 0) or ((b2 + g2 + r2) == 0):
            div = 1

        b = (b1*w1 + b2*w2) / div
        g = (g1*w1 + g2*w2) / div
        r = (r1*w1 + r2*w2) / div

        return np.array([b, g, r])


# A wrapper for warp_pure_pixel_internal
# P1 = H x P
# Input params:
# h = Homography
# pt = tuple of coordinates position for the loop in the big image
#
# Output Params
# u, v = coordinates on the original image
def warp_pure_pt(h, pt):
    v, u = warp_pure_pixel_internal(h, pt[0], pt[1])
    return np.array([u, v])


# A function that calc a new position for the point
# P1 = H x P
# Input params:
# h = Homography
# x = position for the loop in the big image
# y = position for the loop in the big image
#
# Output Params
# u, v = coordinates on the original image
def warp_pure_pixel_internal(h, x, y):
    u = (h[0, 0] * x) + (h[0, 1] * y) + (h[0, 2])
    v = (h[1, 0] * x) + (h[1, 1] * y) + (h[1, 2])
    w = (h[2, 0] * x) + (h[2, 1] * y) + (h[2, 2])
    u /= w
    v /= w
    return v, u


# A function that is used inside the threaded loop to get the new pixels position
# for the target and the main image
#
# Input params:
# args is a support tuple for arguments
# h_inv = Homography -1
# min_u = minimum position in x for the target warped image
# min_v = minimum position in y for the target warped image
# identity = identity matrix for the main image
# main_image = the main image
# blend_type = the blend type: FEATHER, MERGE or AVERAGE
# target_image = image to be warped
# x = position the loop for the big image
# y = position the loop for the big image
#
# Output Params
# big_image = the image that will receive the blend of the other 2 images


def warp_and_blend(big_image, target_image, y, x, args):
    h_inv, min_u, min_v, identity, main_image, blend_type = args

    y -= min_v
    x -= min_u

    v, u = warp_pure_pixel_internal(h_inv, x, y)
    v = math.ceil(v)
    u = math.ceil(u)

    v1, u1 = warp_pure_pixel_internal(identity, x, y)
    v1 = math.ceil(v1)
    u1 = math.ceil(u1)

    y += min_v
    x += min_u

    target_dim = get_dimensions(h_inv, target_image)
    main_dim = get_dimensions(identity, main_image)

    roi = get_roi(target_dim, main_dim)

    target_pt = u, v
    main_pt = u1, v1
    if blend_type == FEATHERING:
        big_image[y, x] = feathering_2point(target_pt, target_image, target_dim, main_pt, main_image, main_dim, roi)
    if blend_type == AVERAGE:
        big_image[y, x] = get_avg_2point(target_pt, target_image, main_pt, main_image)
    if blend_type == MERGE:
        big_image[y, x] = get_one_of(target_pt, target_image, main_pt, main_image)


