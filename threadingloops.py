# !/bin/python
import threading
import cv2
import numpy as np
import imagehelpers as ih


class ThreadLine(threading.Thread):

    def __init__(self, img, obj_out, y, function, args=None):
        threading.Thread.__init__(self)
        self._img = img
        self._obj_out = obj_out
        self._y = y
        self._function = function
        self._args = args

    def run(self):
        img_h, img_w = self._img.shape[:2]
        for x in range(img_w):
            self._function(self._img, self._obj_out, self._y, x, self._args)


class ThreadingLoopsImage:

    def __init__(self, cores, img, obj_out, func, args=None):
        self._cores = cores
        self._func = func
        self._img = img
        self._obj_out = obj_out
        self._args = args

    def execute(self):
        img_h, img_w = self._img.shape[:2]
        threads = []
        for y in range(img_h):

            l_thread = ThreadLine(self._img, self._obj_out, y, self._func, self._args)
            l_thread.start()
            threads.append(l_thread)
            # print threading.active_count()
            if len(threads) == self._cores-1:
                threads = join_threads(threads)

        if not (threads.__sizeof__() == 0):
            join_threads(threads)


def join_threads(threads):
    for t in threads:
        t.join()
    threads = []
    return threads


def test_function(img, obj_out, y, x, args):
    obj_out[y, x] = np.array([255, 255, 0])
    return obj_out

if __name__ == '__main__':
    _image = cv2.imread('self.jpg')
    tli = ThreadingLoopsImage(4, _image, _image, test_function)
    tli.execute()
    ih.show_image(_image)
