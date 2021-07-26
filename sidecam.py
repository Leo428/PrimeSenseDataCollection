import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2 as cv

class SideCam():
    def __init__(self, cam_id) -> None:
        self.sidecam = cv.VideoCapture(cam_id)
        if not self.sidecam.isOpened():
            print('Open Side Cam failed!')
            exit()
        self.sidecam.set(cv.CAP_PROP_FRAME_WIDTH, 320)  # width=1920
        self.sidecam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

## Use 'help' to get more info
# help(dev.set_image_registration_mode)
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    def get_rgb(self):
        ret, bgr = self.sidecam.read()

        if not ret:
            print('Side cam receiving no frames')
            return None
        return bgr

    def close(self):
        self.sidecam.release()

