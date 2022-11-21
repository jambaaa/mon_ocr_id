import cv2

from PIL import Image

import pytesseract

import matplotlib.pyplot as plt

import numpy as np

class OCR:
    def __init__(self, img, original_path, result_path):
        self.img = img
        self.original_path = original_path
        self.result_path = result_path
    
    # def invert_image(self):
    #     self.img = cv2.bitwise_not(self.img)
    #     cv2.imwrite(self.result_path, self.img)
    
    # def binarization(self):
    #     self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite(self.result_path, self.img)
    #     thresh, self.img  = cv2.threshold(self.img, 87, 97, cv2.THRESH_BINARY)
    #     cv2.imwrite(self.result_path, self.img)

    # def noise_removal(self):
    #     kernel = np.ones((1, 1), np.uint8)
    #     self.img = cv2.dilate(self.img, kernel, iterations=1)
    #     kernel = np.ones((1, 1), np.uint8)
    #     self.img = cv2.erode(self.img, kernel, iterations=1)
    #     self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
    #     self.img = cv2.medianBlur(self.img, 3)
    #     cv2.imwrite(self.result_path, self.img)

    # def thin_front(self):
    #     self.img = cv2.bitwise_not(self.img)
    #     kernel = np.ones((2, 2), np.uint8)
    #     self.img = cv2.erode(self.img, kernel, iterations=1)
    #     self.img = cv2.bitwise_not(self.img)
    #     cv2.imwrite(self.result_path, self.img)
    
    # def thick_front(self):
    #     self.img = cv2.bitwise_not(self.img)
    #     kernel = np.ones((2, 2), np.uint8)
    #     self.img = cv2.dilate(self.img, kernel, iterations=1)
    #     self.img = cv2.bitwise_not(self.img)
    #     cv2.imwrite(self.result_path, self.img)

    # def deskew(self):
    #     angle = getSkewAngle(self.img)
    #     self.img = rotateImage(self.img, -1.0 * angle)
    #     cv2.imwrite(self.result_path, self.img)

    def display_done(self):
        dpi = 80
        im_data = plt.imread(self.result_path)
        height, width, depth = im_data.shape
        # height, width = im_data.shape 
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(im_data, cmap='gray')
        plt.show()

# def getSkewAngle(cvImage) -> float:
#     # Prep image, copy, convert to gray scale, blur, and threshold
#     newImage = cvImage.copy()
#     gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9, 9), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     # Apply dilate to merge text into meaningful lines/paragraphs.
#     # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
#     # But use smaller kernel on Y axis to separate between different blocks of text
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
#     dilate = cv2.dilate(thresh, kernel, iterations=2)
#     # Find all contours
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     for c in contours:
#         rect = cv2.boundingRect(c)
#         x,y,w,h = rect
#         cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

#     # Find largest contour and surround in min area box
#     largestContour = contours[0]

#     # --------------------------- TESTING --------------------------
#     # allContourAngles = [cv2.minAreaRect(c)[-1] for c in contours]
#     # angle = sum(allContourAngles) / len(allContourAngles)
#     # middleContour = contours[len(contours) // 2]
#     # angle = cv2.minAreaRect(middleContour)[-1]
#     # largestContour = contours[0]
#     # middleContour = contours[len(contours) // 2]
#     # smallestContour = contours[-1]
#     # angle = sum([cv2.minAreaRect(largestContour)[-1], cv2.minAreaRect(middleContour)[-1], cv2.minAreaRect(smallestContour)[-1]]) / 3
#     # --------------------------- TESTING --------------------------

#     minAreaRect = cv2.minAreaRect(largestContour)
#     cv2.imwrite(result_path, newImage)
#     # Determine the angle. Convert it to the value that was originally used to obtain skewed image
#     angle = minAreaRect[-1]
#     # print("==>", angle)
#     # if angle < 45:
#     #     angle = angle
#     # elif angle < -45:
#     #     angle = 90 + angle
#     # elif angle > -45:
#     #     angle = angle - 105

#     return -1.0 * angle

# # Rotate the image around its center
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return newImage

# initial values
original_path = "data/rotated_id.jpg"
result_path = "process/done.jpg"
image = cv2.imread(original_path)

# optional operation
# image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# OCR step by step
imgObj = OCR(image, original_path, result_path)

# imgObj.thick_front()
# imgObj.invert_image()
# imgObj.binarization()
# imgObj.noise_removal()
# imgObj.thin_front()
# imgObj.deskew()
imgObj.display_done()




# utils

# cv2.imwrite("data/rotated_id.jpg", image)