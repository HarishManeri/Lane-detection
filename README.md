# Lane-detection
Cothon Internship
import cv2
import numpy as np

def detect_lanes(image):
  """
  Detects lanes in an image using OpenCV.

  Args:
    image: A NumPy array representing the image.

  Returns:
    A NumPy array representing the image with lanes drawn on it.
  """

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur to reduce noise
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  # Detect edges using Canny edge detection
  edges = cv2.Canny(blur, 50, 150)

  # Define the region of interest (ROI)
  height, width = image.shape[:2]
  roi = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)

  # Mask the image to only include the ROI
  mask = np.zeros_like(edges)
  cv2.fillPoly(mask, roi, 255)
  masked_edges = cv2.bitwise_and(edges, mask)

  # Apply Hough transform to detect lines
  lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)

  # Draw the detected lines on the image
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

  return image

# Load an image
image = cv2.imread("lane.jpg")

# Detect lanes
lanes_image = detect_lanes(image)

# Display the result
cv2.imshow("Lanes", lanes_image)
cv2.waitKey(0)
