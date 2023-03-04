from __future__ import annotations

import os

import cv2
import numpy as np
import numpy.typing as npt
from dense_feature_detector import DenseFeatureDetector
from skimage import feature


class Visualizer(object):
    def __init__(self, dst_root: str) -> None:
        """Initialize Visualizer class.

        Args:
            dst_root (str): The path to the directory where the visualizations will be saved.
        """
        self.dst_root = dst_root
        os.makedirs(dst_root, exist_ok=True)

    def __call__(self, filename: str) -> None:
        """Generate visualizations of an image.

        Args:
            filename (str): The path to the image file.
        """
        image = cv2.imread(filename)
        try:
            _ = image.shape
        except Exception:
            print("can't read image: {}".format(filename))
        name, _ = os.path.splitext(filename)
        filename_template = "{}/{}_%s.png".format(self.dst_root, name)
        self.gray(image, filename_template)
        self.color_channel(image, filename_template)
        self.gray_gradient(image, filename_template)
        self.color_gradient(image, filename_template)
        self.power_spectrum(image, filename_template)
        self.draw_keypoint(image, filename_template)
        self.draw_hog(image, filename_template)
        self.draw_lbp(image, filename_template)

    def gray(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """Generate grayscale image.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): The path and filename to save the generated image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filename % "gray", gray)

    def color_channel(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """Generate color channel images.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): The path and filename to save the generated images.
        """
        blue = np.zeros_like(image)
        green = np.zeros_like(image)
        red = np.zeros_like(image)
        blue[:, :, 0] = image[:, :, 0]
        green[:, :, 1] = image[:, :, 1]
        red[:, :, 2] = image[:, :, 2]
        cv2.imwrite(filename % "blue", blue)
        cv2.imwrite(filename % "green", green)
        cv2.imwrite(filename % "red", red)

    def gray_gradient(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """Compute and save the grayscale gradient of the input image

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): The path and filename to save the generated images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
        kernel_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], np.float32)
        kernel_xy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]], np.float32)
        grad_x = cv2.filter2D(gray, -1, kernel_x)
        grad_y = cv2.filter2D(gray, -1, kernel_y)
        grad_xy = cv2.filter2D(gray, -1, kernel_xy)
        cv2.imwrite(filename % "gradient_x", grad_x)
        cv2.imwrite(filename % "gradient_y", grad_y)
        cv2.imwrite(filename % "gradient_xy", grad_xy)

    def color_gradient(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """Compute and save the color gradient of the input image

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): A string containing the path and filename of the output image.
        """
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        grad_xy = np.zeros_like(image)
        kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
        kernel_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], np.float32)
        kernel_xy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]], np.float32)
        grad_x[:, :, 0] = cv2.filter2D(image[:, :, 0], -1, kernel_x)
        grad_y[:, :, 0] = cv2.filter2D(image[:, :, 0], -1, kernel_y)
        grad_xy[:, :, 0] = cv2.filter2D(image[:, :, 0], -1, kernel_xy)
        grad_x[:, :, 1] = cv2.filter2D(image[:, :, 1], -1, kernel_x)
        grad_y[:, :, 1] = cv2.filter2D(image[:, :, 1], -1, kernel_y)
        grad_xy[:, :, 1] = cv2.filter2D(image[:, :, 1], -1, kernel_xy)
        grad_x[:, :, 2] = cv2.filter2D(image[:, :, 2], -1, kernel_x)
        grad_y[:, :, 2] = cv2.filter2D(image[:, :, 2], -1, kernel_y)
        grad_xy[:, :, 2] = cv2.filter2D(image[:, :, 2], -1, kernel_xy)
        cv2.imwrite(filename % "gradient_color_x", grad_x)
        cv2.imwrite(filename % "gradient_color_y", grad_y)
        cv2.imwrite(filename % "gradient_color_xy", grad_xy)

    def power_spectrum(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """Compute and save the power spectrum of the input image.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): A string containing the path and filename of the output image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray)
        fft = np.fft.fft2(gray)  # type: ignore
        fft = np.fft.fftshift(fft)  # type: ignore
        pow = np.abs(fft) ** 2
        pow = np.log10(pow)
        Pmax = np.max(pow)  # type: ignore
        pow = pow / Pmax * 255
        pow_image = np.array(np.uint8(pow))
        cv2.imwrite(filename % "power_spectrum", pow_image)

    def draw_keypoint(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """Compute and save the keypoints of the input image using SIFT, AKAZE, and Dense methods.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): A string containing the path and filename of the output image.
        """
        sift_image, rich_sift_image = self.make_sift_image(image)
        akaze_image, rich_akaze_image = self.make_akaze_image(image)
        dense_image, rich_dense_image = self.make_dense_image(image)
        cv2.imwrite(filename % "sift", sift_image)
        cv2.imwrite(filename % "sift_rich", rich_sift_image)
        cv2.imwrite(filename % "akaze", akaze_image)
        cv2.imwrite(filename % "akaze_rich", rich_akaze_image)
        cv2.imwrite(filename % "dense", dense_image)
        cv2.imwrite(filename % "dense_rich", rich_dense_image)

    def make_sift_image(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Detect SIFT keypoints and descriptors of the input image.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: _description_
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_image = image.copy()
        rich_image = image.copy()
        detector = cv2.SIFT_create()
        keypoints = detector.detect(gray)
        color = (255, 255, 0)
        for key in keypoints:
            cv2.circle(
                kp_image, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, color, 1
            )
        cv2.drawKeypoints(
            rich_image,
            keypoints,
            rich_image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        print("num of kps(SIFT) : {:5d}".format(len(keypoints)))
        return kp_image, rich_image

    def make_akaze_image(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Detect AKAZE keypoints and descriptors of the input image.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: _description_
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_image = image.copy()
        rich_image = image.copy()
        detector = cv2.AKAZE_create()
        keypoints = detector.detect(gray)
        color = (255, 255, 0)
        for key in keypoints:
            cv2.circle(
                kp_image, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, color, 1
            )
        cv2.drawKeypoints(
            rich_image,
            keypoints,
            rich_image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        print("num of kps(AKAZE): {:5d}".format(len(keypoints)))
        return kp_image, rich_image

    def make_dense_image(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """
        Converts a 3-channel color image to grayscale and applies dense feature detection
        to generate keypoints. Draws the keypoints on a copy of the original image and on a
        copy of the original image with rich keypoints, and returns them as a tuple.


        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: The original image with keypoints drawn on it and a copy of the original image with rich keypoints.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_image = image.copy()
        rich_image = image.copy()
        detector = DenseFeatureDetector(cv2.SIFT_create(), step=5, scale=5, start=0)
        keypoints, _ = detector.detect(gray)
        color = (255, 255, 0)
        for key in keypoints:
            cv2.circle(
                kp_image, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, color, 1
            )
        cv2.drawKeypoints(
            rich_image,
            keypoints,
            rich_image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        print("num of kps(DENSE): {:5d}".format(len(keypoints)))
        return kp_image, rich_image

    def draw_hog(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """
        Applies histogram of oriented gradients (HOG) feature extraction to an image
        and saves the resulting HOG image to a file.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): A string containing the path and filename of the output image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, hog_image = feature.hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )
        hog_image = np.uint8(hog_image * 255)
        cv2.imwrite(filename % "hog", hog_image)

    def draw_lbp(self, image: npt.NDArray[np.uint8], filename: str) -> None:
        """
        Applies local binary pattern (LBP) feature extraction to an image and saves the
        resulting LBP image to a file.

        Args:
            image (npt.NDArray[np.uint8]): 3channel color image.
            filename (str): A string containing the path and filename of the output image.
        """
        bordered_image = cv2.copyMakeBorder(
            image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        gray = cv2.cvtColor(bordered_image, cv2.COLOR_BGR2GRAY)
        counter = 0
        lbp = 8 * [0]
        lbp_image = np.zeros((gray.shape[0] - 2, gray.shape[1] - 2), dtype=np.uint8)
        for centerY in range(1, gray.shape[0] - 1):
            for centerX in range(1, gray.shape[1] - 1):
                for yy in range(centerY - 1, centerY + 2):
                    for xx in range(centerX - 1, centerX + 2):
                        if (xx != centerX) or (yy != centerY):
                            if gray[centerY, centerX] >= gray[yy, xx]:
                                lbp[counter] = 0
                            else:
                                lbp[counter] = 1
                            counter += 1
                lbp_pix = (
                    lbp[0] * 2 ** 7
                    + lbp[1] * 2 ** 6
                    + lbp[2] * 2 ** 5
                    + lbp[4] * 2 ** 4
                    + lbp[7] * 2 ** 3
                    + lbp[6] * 2 ** 2
                    + lbp[5] * 2 ** 1
                    + lbp[3] * 2 ** 0
                )
                lbp_image[centerY - 1, centerX - 1] = lbp_pix
                counter = 0
        cv2.imwrite(filename % "lbp", lbp_image)
