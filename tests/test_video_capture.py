import unittest
from video_capture import VideoCapture

class TestVideoCaptures(unittest.TestCase):
    def test_cap(self):
        test_vid = "video/abc_2s.mp4"
        # test_vid = "video/test_movie_cc.mp4"
        vc = VideoCapture(video_path=test_vid)
        vc.cap_frames_text()

        self.assertNotEquals(vc.db_captures.count_pframes(), 0)

if __name__ == "__main__":
    unittest.main()