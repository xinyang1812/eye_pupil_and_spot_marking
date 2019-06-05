# -*- coding: UTF-8 -*-
import numpy as np
import os
import cv2
import imutils
from PIL import Image
from pylab import *

"""
This program is the final demo for detecting eye pupil and spot. The main difference between this program and 
the previous version is that in the process of detecting pupil center, the approximate position of pupil is 
located by Hough circle first, then ellipse detection is carried out in this position to find the center coordinate 
of ellipse and return to it.              19.06.05


note: This program only updates the pupil coordinates in the original version, so the coordinates of spot location are not changed.
"""


def read_directory(directory_name, right, light, imagename):

    # Read the file name and return the size and location of the box selected

    for filename in os.listdir(directory_name):
        if filename[-4:] == '.bmp':
            ii = 0
            R = [[] for ii in range(1)]
            L = [[] for ii in range(1)]
            s1, s2, s3 = filename.split('-')
            xr, yr, wr, hr = s2.split(',')
            yr = np.array(yr)
            xr = np.array(xr[1:])
            wr = np.array(wr)
            hr = np.array(hr)
            R[0].append(xr), R[0].append(yr), R[0].append(wr), R[0].append(hr)
            R = np.array(R)
            right.append(R[0])

            xl, yl, wl, hl = s3.split(',')
            hl = np.array(hl[:-4])
            xl = np.array(xl[1:])
            yl = np.array(yl)
            hl = np.array(hl)
            L[0].append(xl), L[0].append(yl), L[0].append(wl), L[0].append(hl)
            L = np.array(L)
            light.append(L[0])

            imagename.append(filename)
    return right, light, imagename

def core(thresh, img_gray):

    # Finding Centroid Coordinates
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    index = [[] for i in range(1)]

    # loop over the contours
    for c in cnts:
        if len(c) > 3:
            # compute the center of the contour
            try:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print (cX, cY)

                index[0].append(cX)
                index[0].append(cY)
                # cv2.namedWindow('core')
                cv2.circle(img_gray, (cX, cY), 1, (255, 255, 255), -1)
                # cv2.imshow('core', img_gray)
                # cv2.waitKey(0)
            except:
                continue
    return index[0]

def cicle(gray):
    # 霍夫变换圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=140, param2=10, minRadius=5, maxRadius=15)
    R = 0
    for circle in circles[0]:
        print(circle[2])
        if int(circle[2]) > R:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            R = r
        gray = cv2.circle(gray, (x, y), r, (255, 0, 0), 1, 8, 0)
        w, h = gray.shape
        # cv2.circle(gray, (x, y), 1, (255, 0, 0), -1)
        # gray = cv2.resize(gray, (h*4, w*4), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('5', gray)
        # cv2.waitKey(100)
    return x,y,r

def test_fitEllipse(img):
    imgray = cv2.Canny(img, 600, 100, 3)  # Canny边缘检测，参数可更改
    # cv2.imshow("0",imgray)
    ret, thresh = cv2.threshold(imgray, 220, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        if len(cnt) > 18:
            S1 = cv2.contourArea(cnt)
            ell = cv2.fitEllipse(cnt)
            S2 = math.pi * ell[1][0] * ell[1][1]
            if (S1 / S2) > 0.2:  # 面积比例，可以更改，根据数据集。。。
                img = cv2.ellipse(img, ell, (255, 0, 0), 2)
                cv2.circle(img, (int(ell[0][0]), int(ell[0][1])), 1, (0,0,0), -1)
                # cv2.imshow('img', img)
    return int(ell[0][0]), int(ell[0][1])

def Right(cropright, right, coreright, map, imagename, directory):

    # Processing the right eye and returning to the pupil center and spot coordinates of the right eye

    img_gray_cropright = cv2.cvtColor(cropright, cv2.COLOR_RGB2GRAY)

    # Finding the centroid coordinates of the pupil（single）
    ii = 0
    index = [[] for ii in range(2)]
    x, y, r = cicle(img_gray_cropright)
    index[0] = int(x) + int(right[i][0])
    index[1] = int(y) + int(right[i][1])

    w = index[0]
    h = index[1]
    img_m = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)
    img_cicle = img_m[h-int(1.5*r):h+int(1.5*r),w-int(1.5*r):w+int(1.5*r)]
    image_height, image_weight = img_cicle.shape[0], img_cicle.shape[1]
    dst = np.zeros((image_height, image_weight, 1), np.uint8)
    for k in range(image_height):
        for j in range(image_weight):
            grayPixel = img_cicle[k][j]
            dst[k][j] = 255 - grayPixel

    ret, thresh2 = cv2.threshold(dst, 220, 255, cv2.cv2.THRESH_BINARY)
    # thresh2 = cv2.resize(thresh2, (image_height * 4, image_weight * 4), interpolation=cv2.INTER_CUBIC)

    ell1, ell2 = test_fitEllipse(thresh2)

    index2 = [[] for j in range(2)]
    index2[0] = ell1 + int(w-int(1.5*r))
    index2[1] = ell2 + int(h-int(1.5*r))

    cv2.circle(map, (index2[0], index2[1]), 1, (0, 255, 255), -1)
    # cv2.imshow('ss',map)
    # cv2.imshow('closing', thresh2)
    # # cv2.imshow('sss', thresh2)
    # cv2.waitKey(0)

    coreright.append(index2)

    # Finding the centroid coordinates of the spot（Multiple）

    # image = Image.fromarray(cv2.cvtColor(cropright, cv2.COLOR_BGR2RGB))
    # imshow(image)
    # xxx = ginput(2)
    # plt.close()
    # index = [[] for ii in range(4)]
    # index[0] = int(xxx[0][0]) + int(right[i][0])
    # index[1] = int(xxx[0][1]) + int(right[i][1])
    # index[2] = int(xxx[1][0]) + int(right[i][0])
    # index[3] = int(xxx[1][1]) + int(right[i][1])
    # coreright.append(index)
    return coreright

def Left(cropleft, left, coreleft, map, imagename, directory):

    # Processing the left eye and returning to the pupil center and spot coordinates of the left eye

    img_gray_cropleft = cv2.cvtColor(cropleft, cv2.COLOR_RGB2GRAY)

    # Finding the centroid coordinates of the pupil（single）
    ii = 0
    index = [[] for ii in range(2)]
    x, y, r = cicle(img_gray_cropleft)
    index[0] = int(x) + int(left[i][0])
    index[1] = int(y) + int(left[i][1])

    w = index[0]
    h = index[1]
    img_m = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)
    img_cicle = img_m[h-int(1.5*r):h+int(1.5*r),w-int(1.5*r):w+int(1.5*r)]
    image_height, image_weight = img_cicle.shape[0], img_cicle.shape[1]
    dst = np.zeros((image_height, image_weight, 1), np.uint8)
    for k in range(image_height):
        for j in range(image_weight):
            grayPixel = img_cicle[k][j]
            dst[k][j] = 255 - grayPixel

    ret, thresh2 = cv2.threshold(dst, 220, 255, cv2.cv2.THRESH_BINARY)

    ell1, ell2 = test_fitEllipse(thresh2)

    index2 = [[] for j in range(2)]
    index2[0] = ell1 + int(w-int(1.5*r))
    index2[1] = ell2 + int(h-int(1.5*r))
    coreleft.append(index)

    cv2.circle(map, (index2[0], index2[1]), 1, (0, 255, 255), -1)
    # cv2.imshow('ss',map)
    # cv2.imshow('closing', thresh2)
    # # cv2.imshow('sss', thresh2)
    # cv2.waitKey(0)

    # Finding the centroid coordinates of the spot（Multiple）

    # image = Image.fromarray(cv2.cvtColor(cropleft, cv2.COLOR_BGR2RGB))
    # imshow(image)
    # xxx = ginput(2)
    # plt.close()
    # index = [[] for ii in range(4)]
    # index[0] = int(xxx[0][0]) + int(left[i][0])
    # index[1] = int(xxx[0][1]) + int(left[i][1])
    # index[2] = int(xxx[1][0]) + int(left[i][0])
    # index[3] = int(xxx[1][1]) + int(left[i][1])
    # coreleft.append(index)
    return coreleft

def text_create(name, msg, directory):

    #Create a new TXT file and write the coordinate information

    full_path = directory + name + '.txt'
    file = open(full_path,'w')
    file.write(msg)
    file.close()
    print('Done')

def text_fix(name, directory, r, l):
    full_path = directory + name + '.txt'
    fp = file(full_path)
    lines = []
    for line in fp:
        lines.append(line)
    fp.close()
    del lines[1]
    lines.insert(1, r)  # 在第二行插入
    del lines[3]
    lines.insert(3,l)
    fp = file(full_path, 'w')
    for ii in lines:
      fp.write(ii)
    fp.close()

if __name__=='__main__':
    right = []
    left = []
    imagename = []
    # directory = 'E:/OverMission/1086-190402/te-st/tes-t9/test/'
    directory = 'E:/OverMission/1086-190402/te-st/t-15/'

    right, left, imagename = read_directory(directory, right, left, imagename)
    image = []

    i = 0
    for x in imagename:
        print('+++++++++++++++++++++++++++++++++++')
        print(x + '  begin')
        map = cv2.imread(directory + x)
        cropright = map[int(right[i][1]):(int(right[i][1]) + int(right[i][3])),
                    int(right[i][0]):(int(right[i][0]) + int(right[i][2]))]
        croplight = map[int(left[i][1]):(int(left[i][1]) + int(left[i][3])),
                    int(left[i][0]):(int(left[i][0]) + int(left[i][2]))]

        coreright = []
        coreleft = []
        coreright = Right(cropright, right, coreright, map, x, directory)
        coreleft = Left(croplight, left, coreleft, map, x, directory)

        # Used to generate new text
        # text_create(x[:-4], 'r-spot: ' + str(coreright[1]) + '\n' + 'r-pupil: ' + str(coreright[0]) +'\n'+ 'l-spot: ' + str(coreleft[1]) +'\n'+ 'l-pupil: ' + str(coreleft[0]), directory)

        # Adding content to existing text
        text_fix(x[:-4], directory, 'r-pupil: ' + str(coreright[0]) +'\n', 'l-pupil: ' + str(coreleft[0]) +'\n')

        print(x + '  done!')

        i = i + 1
        image.append(cv2.imread(directory + x, 0))
