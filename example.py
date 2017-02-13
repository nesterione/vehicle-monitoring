# Following example shows how to detect cars on video


from detect.classifier import CarSimpleDetector
from detect.classifier import SimpleHaarClassifier
import numpy as np
import cv2


class BackgoundSubstractor:

    def __init__(self):
        """
        :param detection_algorithm: Takes frame and return list of found objects
        """
        # createbackgroundSubstractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # fgbg.setBackgroundRatio(1.0)
        self.fgbg.setComplexityReductionThreshold(100)
        self.fgbg.setHistory(10)
        self.fgbg.setNMixtures(2)
        self.fgbg.setDetectShadows(True)
        self.fgbg.setShadowValue(2)
        # fgbg.setShadowThreshold(0.5)
        # fgbg.setShadowValue(True)
        # fgbg.setVarInit(10)

    def apply(self, frame):
        fgmask = self.fgbg.apply(frame, None, 0.002)
        return cv2.bitwise_and(frame, frame, mask=fgmask)



def to_gray_processor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def resize_processor(frame):
    #f = np.ndarray(shape=(720, 405), dtype=np.uint8)
    frame = cv2.resize(src=frame, dsize=(720, 405))
    # frame = cv2.resize(src=frame, dsize=(360, 200))
    return frame


if __name__ == '__main__':

    s = cv2.imread("/home/igor/TRAIN/cap.png",-1)
    s = cv2.resize(src=s, dsize=(50, 50))
    print(s.shape)


    detector = CarSimpleDetector(detection_algorithm=SimpleHaarClassifier())
    detector.add_processor(to_gray_processor)
    detector.add_processor(resize_processor)
    detector.add_processor(BackgoundSubstractor().apply)
    detector.add_processor((lambda frame: 255-frame))

    cap = cv2.VideoCapture('part-reilway.mp4')
    end_frame = 600
    while (end_frame>0):
        end_frame -= 1

        ret, original_frame = cap.read()

        # frame = detector.process(original_frame)
        original_frame = resize_processor(original_frame)
        # print(.shape)

        for (x, y, w, h) in detector.detect(original_frame):
            #original_frame = cv2.rectangle(original_frame, (x, y), (x + w, y + h), (200,200, 0), 2)
            #s.copyTo(original_frame( cv2.rectangle(x, y, s.cols, s.rows)))
            y = y - 25
            y+=180

            if original_frame[y:y + s.shape[0], x:x + s.shape[1]].shape == (50,50,3):
                for c in range(0, 3):
                    original_frame[y:y+ s.shape[0], x:x+ s.shape[1], c] = s[:, :, c] * (s[:, :, 3] / 255.0) + original_frame[y:y + s.shape[0], x:x + s.shape[1], c] * (1.0 - s[:, :, 3] / 255.0)

        cv2.imshow('frame', original_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()