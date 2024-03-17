"""
VideoCapture
Video capture from file and live camera
"""
import cv2
from captures import Captures
import numpy as np
import pickle

# class logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('video_capture')

class VideoCapture:
    def __init__(self, video_path: str=None):
        self.video_path = video_path
        self.cap = None
        self.frame = None
        self.gray_frame = None
        self.db_captures = Captures()

    def cap_frames(self):
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            # use cam
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logger.error("cv2.VideoCapture could not open video")
            raise IOError

        # Read a frame from the video source
        ret, self.frame = self.cap.read()

        # If the frame was not retrieved successfully, return None
        if not ret:
            logger.error("Cannot read video frame")
            raise IOError

        # convert frame to keras tensor grayscale
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Convert to NumPy array
        self.gray_frame = np.asarray(self.gray_frame)
        # Expand dimension for channel (Keras expects grayscale as single-channel)
        self.gray_frame = np.expand_dims(self.gray_frame, axis=-1)
        # Convert to float32 (common data type for Keras tensors)
        self.gray_frame = self.gray_frame.astype('float32')

        # save frame in DB
        try:
            # pickle encode
            pframe = pickle.dumps(self.gray_frame)
            self.db_captures.insert_pframe_tensor(pframe)
        except Exception as err:
            logger.error(f"Insert of pframe failed\n{err}")
            raise

