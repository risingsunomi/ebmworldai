import unittest
from video_capture import VideoCapture

class TestVideoCaptures(unittest.TestCase):
    def test_cap(self):
        test_vid = "video/abc_2s.mp4"
        vc = VideoCapture(video_path=test_vid)

        with self.assertRaises(Exception):
            vc.cap_frames()

if __name__ == "__main__":
    unittest.main()