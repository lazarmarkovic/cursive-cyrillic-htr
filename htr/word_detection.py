import os
import sys
import numpy as np
import cv2
import math


def detect(image):
    # Image pre-processing - blur, edges, threshold, closing
    blurred = cv2.GaussianBlur(image, (5, 5), 18)
    edges = _edge_detect(blurred)
    ret, edges = cv2.threshold(edges, 80, 255, cv2.THRESH_BINARY)
    before_dilate = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    bw_image = cv2.dilate(before_dilate, kernel, iterations=4)

    boxes = _text_detect(bw_image, before_dilate)

    rois = []
    for (x, y, w, h, i) in boxes:
        roi = image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
        rois.append(roi)

    return rois


def extract_and_save_data(boxes, original_image):
    count = 0
    for (x, y, w, h, i) in boxes:
        roi = original_image[y:y+h, x:x+w]
        cv2.imwrite("./training_data/velika1/" + str(count) + ".jpg", roi)
        count += 1


def _edge_detect(im):
    """
    Edge detection
    The Sobel operator is applied for each image layer (RGB)
    """
    return np.max(np.array([_sobel(im[:, :, 0]), _sobel(im[:, :, 1]), _sobel(im[:, :, 2])]), axis=0)


def _sobel(channel):
    """ The Sobel Operator"""
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def _text_detect(img, before_dilate):
    """ Text detection using contours """

    # Finding contours
    mask = np.zeros(img.shape, np.uint8)
    im2, cnt, hierarchy = cv2.findContours(
        np.copy(img), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Variables for contour index and words' bounding boxes
    index = 0
    boxes = []

    boxes_with_area = np.empty((0, 5), float)

    # CCOMP hierarchy: [Next, Previous, First Child, Parent]
    # cv2.RETR_CCOMP - contours into 2 levels
    # Go through all contours in first level
    while (index >= 0):
        x, y, w, h = cv2.boundingRect(cnt[index])
        # Get only the contour
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = before_dilate[y:y+h, x:x+w]

        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)

        # Limits for text (white pixel ratio, width, height)
        if r > 0.1 and 2000 > w > 10 and 1600 > h > 10 and h/w < 3 and w/h < 10:
            a = [x, y, w, h]
            area = _rectangle_area(a)
            box_with_area = np.array([x, y, w, h, area])
            boxes_with_area = np.vstack((boxes_with_area, box_with_area))

        # Index of next contour
        index = hierarchy[0][index][0]

    boxes = _reject_outliers(boxes_with_area)

    # Sort

    def keyy(val):
        # return math.sqrt(val[0]**2 + val[1]**2) + val[1]
        # return val[1]**2 + math.sqrt(val[0]**2 + val[1]**2)
        return val[1]

    boxes = boxes.tolist()
    boxes.sort(key=keyy)

    return boxes


def _rectangle_area(c):
    return c[2]*c[3]


def _reject_outliers(data, m=3):
    data[:, 4] = (data[:, 4] - np.amin(data[:, 4])) / \
        (np.amax(data[:, 4]) - np.amin(data[:, 4]))

    data = data[abs(data[:, 4] - np.mean(data[:, 4])) < m * np.std(data[:, 4])]
    return data
