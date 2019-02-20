import cv2 as cv
import numpy as np
import sys
import math


def detect(path):
    img_edges = _find_edges(path)
    lines = _find_four_lines(img_edges)
    corners = _find_four_corners(lines)

    original = cv.imread(path)
    cropped_image = _persp_transform(original, corners)

    return cropped_image


def _find_edges(path):
    src = cv.imread(path)

    img_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #img_blured = cv.medianBlur(img_gray, 43)
    #edges = cv.Canny(img_blured, 10, 50, apertureSize=3)
    img_blured = cv.medianBlur(img_gray, 101)
    edges = cv.Canny(img_blured, 10, 50, apertureSize=3)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilated = cv.dilate(edges, kernel)

    img_with_border = cv.copyMakeBorder(dilated, 10, 10, 10, 10,
                                        cv.BORDER_CONSTANT, value=[255, 255, 255])

    return img_with_border


def _find_four_lines(img):
    lines = cv.HoughLines(img, 1, math.pi/180.0, 250, np.array([]), 0, 0)

    if lines is not None:
        average_v_angle_dev = _find_average_vertical_angle_deviation(lines)
        lines = [_find_top(lines, img, average_v_angle_dev),
                 _find_bottom(lines, img, average_v_angle_dev),
                 _find_left(lines, img, average_v_angle_dev),
                 _find_right(lines, img, average_v_angle_dev)]
        return lines
    return None


def _find_four_corners(lines):
    return np.array(_get_corners(lines))


def _persp_transform(img, s):
    s_points = np.array([s[0], s[2], s[3], s[1]])

    height = max(np.linalg.norm(s_points[0] - s_points[1]),
                 np.linalg.norm(s_points[2] - s_points[3]))
    width = max(np.linalg.norm(s_points[1] - s_points[2]),
                np.linalg.norm(s_points[3] - s_points[0]))

    t_points = np.array([[0, 0],
                         [0, height],
                         [width, height],
                         [width, 0]], np.float32)

    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)

    M = cv.getPerspectiveTransform(s_points, t_points)
    return cv.warpPerspective(img, M, (int(width), int(height)))


# --------------------------------------------------------------------------

def _get_corners(polar):
    result = [_calc_point(polar, 0, 2),
              _calc_point(polar, 0, 3),
              _calc_point(polar, 1, 2),
              _calc_point(polar, 1, 3)]

    return result


def _calc_point(polar, index1, index2):
    aa = np.empty((0, 2), float)
    bb = np.empty((0, 1), float)

    rho = polar[index1][0]
    theta = polar[index1][1]
    a = math.cos(theta)
    b = math.sin(theta)

    aa = np.vstack([aa, [a, b]])
    bb = np.append(bb, rho)

    # ---------------------

    rho = polar[index2][0]
    theta = polar[index2][1]
    a = math.cos(theta)
    b = math.sin(theta)

    aa = np.vstack([aa, [a, b]])
    bb = np.append(bb, rho)

    t = np.linalg.solve(aa, bb)
    # if index1 == 0 and index2 == 2:
    #    return [int(t[0]+60), int(t[1]+60)]
    # if index1 == 0 and index2 == 3:
    #    return [int(t[0]-60), int(t[1]+60)]
    # if index1 == 1 and index2 == 2:
    #    return [int(t[0]+60), int(t[1]-60)]
    # if index1 == 1 and index2 == 3:
    #    return [int(t[0]-60), int(t[1]-60)]

    return (int(t[0]), int(t[1]))


def _find_average_vertical_angle_deviation(lines):
    a, b, c = lines.shape

    theta_interval1 = [0, 20]
    theta_interval2 = [160, 180]

    theta_sum1 = 0
    theta_sum2 = 0

    count1 = 0
    count2 = 0

    for i in range(a):
        theta = lines[i][0][1]*180/math.pi
        if (theta > theta_interval1[0] and theta < theta_interval1[1]):
            theta_sum1 += theta
            count1 += 1

        if (theta > theta_interval2[0] and theta < theta_interval2[1]):
            theta_sum2 += theta
            count2 += 1

    average_angle1 = theta_sum1/count1

    average_angle2 = 180 - theta_sum2/count2

    return (average_angle1 + average_angle2)/2


def _find_top(lines, img, average_v_angle_dev):
    a, b, c = lines.shape
    h, w = img.shape

    max_rho = 0
    found_theta = 0
    max_search = h/2

    #def_angle = 90 + average_v_angle_dev
    def_angle = 90

    theta_interval = [def_angle-1, def_angle+1]

    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]*180/math.pi
        if (rho > max_rho and rho < max_search):

            if (theta > theta_interval[0] and theta < theta_interval[1]):
                max_rho = rho
                found_theta = theta

    return (max_rho, found_theta*math.pi/180)


def _find_bottom(lines, img, average_v_angle_dev):
    a, b, c = lines.shape
    h, w = img.shape

    min_rho = h
    found_theta = 0
    max_search = h/2

    #def_angle = 90 + average_v_angle_dev
    def_angle = 90

    theta_interval = [def_angle-1, def_angle+1]

    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]*180/math.pi
        if (rho < min_rho and rho > max_search):

            if (theta > theta_interval[0] and theta < theta_interval[1]):
                min_rho = rho
                found_theta = theta

    return (min_rho, found_theta*math.pi/180)


def _find_left(lines, img, average_v_angle_dev):
    a, b, c = lines.shape
    h, w = img.shape

    max_rho = 0
    found_theta = 0
    max_search = w/2

    #def_angle1 = 0 + average_v_angle_dev
    #def_angle2 = 180 - average_v_angle_dev

    def_angle1 = 0
    def_angle2 = 180

    theta_interval1 = [def_angle1-2, def_angle1+2]
    theta_interval2 = [def_angle2-2, def_angle2+2]

    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]*180/math.pi

        if (rho > max_rho and rho < max_search):

            if ((theta > theta_interval1[0] and theta < theta_interval1[1]) or
                    (theta > theta_interval2[0] and theta < theta_interval2[1])):
                max_rho = rho
                found_theta = theta

    return (max_rho, found_theta*math.pi/180)


def _find_right(lines, img, average_v_angle_dev):
    a, b, c = lines.shape
    h, w = img.shape

    min_rho = w
    found_theta = 0
    negative = False

    min_search = w/2

    #def_angle1 = 0 + average_v_angle_dev
    #def_angle2 = 180 - average_v_angle_dev

    def_angle1 = 0
    def_angle2 = 180

    theta_interval1 = [def_angle1-2, def_angle1+2]
    theta_interval2 = [def_angle2-2, def_angle2+2]

    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]*180/math.pi

        if (abs(rho) < min_rho and abs(rho) > min_search):
            if ((theta > theta_interval1[0] and theta < theta_interval1[1]) or
                    (theta > theta_interval2[0] and theta < theta_interval2[1])):
                min_rho = abs(rho)
                negative = rho < 0

                found_theta = theta

    return ((-1 if negative else 1) * min_rho, found_theta*math.pi/180)
