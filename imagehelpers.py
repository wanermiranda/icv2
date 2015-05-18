import matplotlib.pyplot as plt
import cv2
import numpy as np
import threadingloops as tl
import math


def determinant(homography):
    return (homography[0, 1] * homography[1, 1] * homography[2, 2]) - (
        homography[2, 0] * homography[1, 1] * homography[0, 2])


# Support function to save images to files
def save_image(img, name):
    plt.imshow(img)
    plt.savefig(name)


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


# Support function to calculate the mean squared error
def get_mse(query, target):
    (m, n) = query.shape[:2]
    if (m == 3) and (n == 256):
        query = query[0:3, 0:256]
        target = target[0:3, 0:256]
    # (m, n) = query.shape[:2]
    # print str(m) + ", " + str(n)
    sums = np.power(np.array(query) - np.array(target), 2).sum()
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


def apply_median(warped_img, skip_zeros=True):
    print "Median"
    tl.ThreadingLoopsImage(4, warped_img, warped_img, get_median_for, skip_zeros).execute()
    return warped_img


def get_median_for(img, img_out, y, x, skip_zeros=True):
    if (img[y, x] == np.array([0, 0, 0])).all() or not skip_zeros:
        b = g = r = 0
        max_h, max_w = img.shape[:2]
        pixels = 0
        for x1 in range(x-1, x+2):
            for y1 in range(y-1, y+2):
                if (y1 < max_h) and (x1 < max_w):
                    if (not (x1 == x) and (y1 == y)) or not skip_zeros:
                        b += img[y1, x1][0]
                        g += img[y1, x1][1]
                        r += img[y1, x1][2]
                        pixels += 1
        b /= pixels
        g /= pixels
        r /= pixels

        img_out[y, x] = np.array([b, g, r])


def warp_pure_pt(h, pt):
    v, u = warp_pure_pixel_internal(h, pt[0], pt[1])
    return np.array([u, v])


def warp_pure_pixel_internal(h_inv, x, y):
    u = (h_inv[0, 0] * x) + (h_inv[0, 1] * y) + (h_inv[0, 2])
    v = (h_inv[1, 0] * x) + (h_inv[1, 1] * y) + (h_inv[1, 2])
    w = (h_inv[2, 0] * x) + (h_inv[2, 1] * y) + (h_inv[2, 2])
    u /= w
    v /= w
    return v, u


# because we are using the inverse homography, the output image will be the input
# H_INV x P' = P from P' = H x P
def warp_pure_pixel(warp_image, out_image, y, x, args):
    h_inv, min_u, min_v, new_h, new_w = args
    orig_h, orig_w = out_image.shape[:2]

    v, u = warp_pure_pixel_internal(h_inv, x, y)
    u -= min_u
    v -= min_v

    # u = math.ceil(u)
    # v = math.ceil(v)
    if (u < orig_w) and (v < orig_h) and (u >= 0) and (v >= 0):
        warp_image[y, x] = out_image[v, u]
