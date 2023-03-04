# source from http://ni4muraano.hatenablog.com/entry/2018/09/14/232746
from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


class DenseFeatureDetector:
    def __init__(self, detector: cv2.SIFT, step: int, scale: int, start: int) -> None:
        """
        Initializes the DenseFeatureDetector object.

        Args:
            detector (cv2.SIFT): A SIFT detector object to compute SIFT features.
            step (int): The step size between each dense keypoint.
            scale (int): The scale of the dense keypoints.
            start (int): The starting offset for the dense keypoints.
        """
        self._detector = detector
        self._step = step
        self._scale = scale
        self._start = start

    def detect(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[list[cv2.KeyPoint], npt.NDArray[np.float32]]:
        """Detects dense keypoints on an image and computes SIFT features on those keypoints.

        Args:
            image (npt.NDArray[np.uint8]): The input image.

        Returns:
            tuple[list[cv2.KeyPoint], npt.NDArray[np.float32]]: A tuple containing a list of detected keypoints and their corresponding feature descriptors.
        """
        # Convert image to gray if it is color
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Create dense keypoints
        keypoints = self._create_keypoints(gray_image)
        _, features = self._detector.compute(image, keypoints)
        return keypoints, features

    def _create_keypoints(
        self, gray_image: npt.NDArray[np.uint8]
    ) -> list[tuple[cv2.KeyPoint, ...]]:
        """Creates dense keypoints on a grayscale image.

        Args:
            gray_image (npt.NDArray[np.uint8]): The input grayscale image.

        Returns:
            list[tuple[cv2.KeyPoint, ...]]: A list of dense keypoints.
        """
        keypoints = []
        rows, cols = gray_image.shape
        for y in range(self._start, rows, self._step):
            for x in range(self._start, cols, self._step):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self._scale))
        return keypoints
