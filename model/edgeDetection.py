#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import logging

# Configure logging (you can customize this)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EdgeDetection:
    """
    A class for performing edge detection on images using the Canny edge detector.
    """

    def auto_canny(self, image, sigma=0.33):
        """
        Automatically determines the lower and upper thresholds for the Canny edge detector
        based on the median pixel value of the image.

        Args:
            image (numpy.ndarray): The input image (grayscale).
            sigma (float, optional): A parameter that controls how wide the thresholds
                                  are spread from the median. Defaults to 0.33.

        Returns:
            numpy.ndarray: The image with edges detected.
        """
        try:
            # Compute the median of the single channel pixel intensities
            med = np.median(image)

            # Apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * med))
            upper = int(min(255, (1.0 + sigma) * med))  # Ensure upper bound is within 0-255
            edged = cv2.Canny(image, lower, upper)

            return edged

        except Exception as e:
            logging.error(f"Error in auto_canny: {e}")
            return None

    def show_image_edge_comparison(self, image_path):
        """
        Reads an image, applies different Canny edge detection methods, and displays the results.

        Args:
            image_path (str): The path to the input image.
        """
        try:
            # Read the original image
            orig_image = cv2.imread(image_path)
            if orig_image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

            # Apply Canny edge detection with different thresholds
            wide_edges = cv2.Canny(blurred_image, 10, 200)
            tight_edges = cv2.Canny(blurred_image, 225, 250)
            auto_edges = self.auto_canny(blurred_image)

            # Display the original image and the edge detection results
            cv2.imshow("Original", orig_image)
            cv2.imshow("Edge Comparison", np.hstack([wide_edges, tight_edges, auto_edges]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
        except cv2.error as cv_error:
            logging.error(f"OpenCV error: {cv_error}")
        except Exception as e:
            logging.error(f"Error in show_image_edge_comparison: {e}")


if __name__ == "__main__":
    try:
        image_path = input("Enter the image path: ")
        edge_detector = EdgeDetection()
        edge_detector.show_image_edge_comparison(image_path)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")