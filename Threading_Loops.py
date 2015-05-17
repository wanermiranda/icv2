# !/bin/python
import threading


class ThreadLine(threading.Thread):

    def __init__(self, img, y, function):
        threading.Thread.__init__(self)
        self._img = img
        self._y = y
        self._function = function

    def run(self):
        img_h, img_w = self._img.shape[:2]
        for x in range(img_w):
            self._function(self._img, self._y, x)


class ThreadingLoopsImage:

    def __init__(self, cores, img, func):
        self._cores = cores
        self._func = func
        self._img = img

    def execute(self):
        img_h, img_w = self._img.shape[:2]
        threads = []
        for y in range(img_h):

            l_thread = ThreadLine(self._img, y, self._func)
            l_thread.start()
            threads.append(l_thread)

            print "Line :" + str(y)

            if threads.__sizeof__() == self._cores:
                threads = join_threads(threads)

        if not (threads.__sizeof__() == 0):
            join_threads(threads)


def join_threads(threads):
    for t in threads:
        t.join()
    threads = []
    return threads


if __name__ == '__main__':
    print ThreadingLoopsImage(4)