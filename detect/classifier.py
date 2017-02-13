import cv2

class CarSimpleDetector:
    '''A car detector. This class '''

    def __init__(self, detection_algorithm):
        """
        :param detection_algorithm: Takes frame and return list of found objects
        """
        self.processors = []
        self.detection_algorithm = detection_algorithm


    def add_processor(self, processor):
        """
        :param processor: function which takes frame do transformation and return changed frame
        :return: changed frame
        """
        self.processors+=[processor]
        return self

    def process(self, frame):
        for processor in self.processors:
            frame = processor(frame)
        return frame

    def detect(self, frame):
        """
        This function return  list of rectangles. Tuple of (x, y, w, h)
        where x,y - position of left top point of rectangle
        w - width
        h - height
        :param frame: numpy array which represents image
        :return: List of tuples (x, y, w, h), if nothing found return empty list
        """
        frame = self.process(frame)
        return self.detection_algorithm.detect(frame)



class SimpleHaarClassifier:

    def __init__(self):
        self.object_cascade = cv2.CascadeClassifier('./detect/cascades/simple_haar_00.xml')


    def detect(self, frame):
        return self.object_cascade.detectMultiScale(frame[180:], 1.6, 25, minSize=(40, 40), maxSize=(80, 80))