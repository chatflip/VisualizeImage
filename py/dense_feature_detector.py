# source from http://ni4muraano.hatenablog.com/entry/2018/09/14/232746

import cv2


class DenseFeatureDetector(object):
    def __init__(self, detector, step, scale, start):
        self._detector = detector
        self._step = step
        self._scale = scale
        self._start = start
        pass

    def detect(self, image):
        # Convert image to gray if it is color
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Create dense keypoints
        keypoints = self._create_keypoints(gray_image)
        _, features = self._detector.compute(image, keypoints)
        return keypoints, features

    def _create_keypoints(self, gray_image):
        keypoints = []
        rows, cols = gray_image.shape
        for y in range(self._start, rows, self._step):
            for x in range(self._start, cols, self._step):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self._scale))
        return keypoints
