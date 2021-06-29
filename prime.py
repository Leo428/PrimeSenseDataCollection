import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

class PrimeSense():
    def __init__(self) -> None:
        openni2.initialize("/home/abhigupta/Downloads/OpenNI-Linux-x64-2.2/Redist")     # can also accept the path of the OpenNI redistribution

        ## Register the device
        self.dev = openni2.Device.open_any()
        ## Create the streams stream
        self.rgb_stream = self.dev.create_color_stream()
        ## Check and configure the depth_stream -- set automatically based on bus speed
        print('The rgb video mode is', self.rgb_stream.get_video_mode()) # Checks rgb video configuration
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))

    def start_stream(self):
        ## Start the streams
        self.rgb_stream.start()

## Use 'help' to get more info
# help(dev.set_image_registration_mode)
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    def get_rgb(self):
        bgr   = np.frombuffer(self.rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
        rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        return rgb

    def close(self):
        self.rgb_stream.stop()
        openni2.unload()

