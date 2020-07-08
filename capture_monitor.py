from mss import mss
import cv2
from PIL import Image
import numpy as np
from threading import Lock

from PIL import Image
from Xlib import X
import ewmh

class CaptureMonitor:
    MODE_CROP, MODE_WINDOW = list(range(2))
    def __init__(self, bb):
        self.lock = Lock()
        self.screen_capture = mss()
        self.ewmh_capture = ewmh.EWMH()

        self.current_window = None
        self.current_monitor = {'left': None, 'top': None, 'width': None, 'height': None}

        self.mode = CaptureMonitor.MODE_CROP
        self.change_coords(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])

        self.max_resolution = 920
        self.min_resolution = 320

    def screen_resize(self, im, size_axis):
        if im.shape[0] > im.shape[1]:
            h = size_axis
            w = h * (im.shape[1] / im.shape[0])
        else:
            w = size_axis
            h = w * (im.shape[0] / im.shape[1])

        h = int(h)
        w = int(w)
        return cv2.resize(im, (w, h))

    def get_frame(self):
        with self.lock:
            if self.mode == CaptureMonitor.MODE_WINDOW and self.current_window is not None:
                geo = self.current_window.get_geometry()
                raw_im = self.current_window.get_image(0, 0, geo.width, geo.height, X.ZPixmap, 0xffffffff)
                im = Image.frombytes("RGB", (geo.width, geo.height), raw_im.data, "raw", "BGRX")
            else:
                im = Image.frombytes('RGB', self.size, self.screen_capture.grab(self.current_monitor).rgb)
            
            im = np.array(im) # RGB

            if im.shape[0] < self.min_resolution or im.shape[1] < self.min_resolution:
                return self.screen_resize(im, self.min_resolution)

            elif im.shape[0] > self.max_resolution or im.shape[1] > self.max_resolution:
                return self.screen_resize(im, self.max_resolution)
            return im


    def get_windows(self):
        windows = []
        for w in self.ewmh_capture.getClientList():
            name = w.get_wm_name().strip()
            if name == '':
                continue

            windows.append(name)

        return ['Active window'] + windows

    def set_window(self, name):
        with self.lock:
            if name == 'Active window':
                self.current_window = self.ewmh_capture.getActiveWindow()
                return

            for w in self.ewmh_capture.getClientList():
                if name == w.get_wm_name().strip():
                    self.current_window = w
                    return

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode

    def change_coords(self, left, top, width, height):
        with self.lock:
            left = int(left)
            top = int(top)
            width = int(width)
            height = int(height)

            self.size = width, height
            self.current_monitor.update({'left': left, 'top': top, 'width': self.size[0], 'height': self.size[1]})


