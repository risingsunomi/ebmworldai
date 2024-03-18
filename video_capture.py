"""
VideoCapture
Video capture from file and live camera
Extracts text but maybe other objects
Useful for creating a testing dataset for the VAE
"""
import cv2
from captures import Captures
import numpy as np
import pickle
from PIL import Image
from pytesseract import pytesseract

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

    def extract_frame_text(self):
        """
        Extract any text picked up by OpenCV and change image to text with pytesseract
        """
        ex_gf = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(ex_gf, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(
            dilation,
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )

        cframe = self.frame.copy()
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = cframe[y:y + h, x:x + w]
            frame_text = pytesseract.image_to_string(cropped, config=("-l eng --oem 1 --psm 7"))
            logger.info(f"Text from Frame: {frame_text}")

    def cap_frames(self):
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            # use cam
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logger.error("cv2.VideoCapture could not open video")
            raise IOError
        
        while True:
            # Read a frame from the video source
            ret, self.frame = self.cap.read()

            # loop from ret
            while ret:
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

                self.extract_frame_text()

                cv2.imshow("Cap Frame", self.frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                ret, self.frame = self.cap.read()
            
            break

        self.cap.release()
        cv2.destroyAllWindows()
            

