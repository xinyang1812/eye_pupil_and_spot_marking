# -*- coding: UTF-8 -*-
import numpy as np
import os
import cv2
import imutils

"""
This program will find the coordinates of the pupil center and the spot center of the eyeball 
in the calibration box positioning area and return to the text.

Note: It is required that the imported folder must be in a uniform format below the image.   19.06.03
"""

def read_directory(directory_name, right, light, imagename):
    # Read the file name and return the size and location of the box selected

    for filename in os.listdir(directory_name):
        R = [[] for i in range(1)]
        L = [[] for i in range(1)]
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
                cv2.circle(img_gray, (cX, cY), 1, (0, 0, 213), -1)
                # cv2.imshow('core', img_gray)
                # cv2.waitKey(0)
            except:
                continue
    return index[0]


def cicle(gray):
    # 霍夫变换圆检测
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=10, minRadius=1, maxRadius=15)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=10, minRadius=1, maxRadius=5)
    # 输出返回值，方便查看类型
    print(circles)
    # 输出检测到圆的个数
    print(len(circles[0]))
    print('-------------我是条分割线-----------------')
    # 根据检测到圆的信息，画出每一个圆
    for circle in circles[0]:
        # 圆的基本信息
        print(circle[2])
        # 坐标行列(就是圆心)
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
        gray = cv2.circle(gray, (x, y), r, (255, 0, 0), 1, 8, 0)
        # cv2.circle(img_gray_cropleft, (x, y), 1, (0, 0, 213), -1)
    # 显示新图像
    cv2.imshow('5', gray)
    cv2.waitKey(0)


def Right(cropright, right, coreright, map):

    # Processing the right eye and returning to the pupil center and spot coordinates of the right eye

    img_gray_cropright = cv2.cvtColor(cropright, cv2.COLOR_RGB2GRAY)
    # Finding the centroid coordinates of the spot（Multiple）
    ret, thresh1 = cv2.threshold(img_gray_cropright, 200, 255, cv2.cv2.THRESH_BINARY)

    index = core(thresh1, cropright)

    if len(index) == 4:
        index[0] = index[0] + int(right[i][0])
        index[1] = index[1] + int(right[i][1])
        index[2] = index[2] + int(right[i][0])
        index[3] = index[3] + int(right[i][1])
        coreright.append(index)
    elif len(index) == 2:
        index[0] = index[0] + int(right[i][0])
        index[1] = index[1] + int(right[i][1])
        coreright.append(index)

    img_info = img_gray_cropright.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    dst = np.zeros((image_height, image_weight, 1), np.uint8)
    for k in range(image_height):
        for j in range(image_weight):
            grayPixel = img_gray_cropright[k][j]
            dst[k][j] = 255 - grayPixel

    # Finding the centroid coordinates of the pupil（single）
    ret, thresh2 = cv2.threshold(dst, 230, 255, cv2.cv2.THRESH_BINARY)

    index = core(thresh2, cropright)

    index[0] = index[0] + int(right[i][0])
    index[1] = index[1] + int(right[i][1])
    coreright.append(index)

    return coreright

def Left(cropleft, left, coreleft, map):

    # Processing the left eye and returning to the pupil center and spot coordinates of the left eye

    img_gray_cropleft = cv2.cvtColor(cropleft, cv2.COLOR_RGB2GRAY)

    # Finding the centroid coordinates of the spot（Multiple）
    ret, thresh1 = cv2.threshold(img_gray_cropleft, 100, 255, cv2.cv2.THRESH_BINARY)

    cicle(img_gray_cropleft)

    index = core(thresh1, cropleft)

    if len(index) == 4:
        index[0] = index[0] + int(left[i][0])
        index[1] = index[1] + int(left[i][1])
        index[2] = index[2] + int(left[i][0])
        index[3] = index[3] + int(left[i][1])
        coreleft.append(index)
    elif len(index) == 2:
        index[0] = index[0] + int(left[i][0])
        index[1] = index[1] + int(left[i][1])
        coreleft.append(index)


    img_info = img_gray_cropleft.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    dst = np.zeros((image_height, image_weight, 1), np.uint8)
    for k in range(image_height):
        for j in range(image_weight):
            grayPixel = img_gray_cropleft[k][j]
            dst[k][j] = 255 - grayPixel

    # Finding the centroid coordinates of the pupil（single）
    ret, thresh2 = cv2.threshold(dst, 230, 255, cv2.cv2.THRESH_BINARY)

    index = core(thresh2, cropleft)
    index[0] = index[0] + int(left[i][0])
    index[1] = index[1] + int(left[i][1])
    coreleft.append(index)
    #显示整体效果图
    # cv2.namedWindow('ds')
    # cv2.imshow('ds', map)
    # cv2.waitKey(100)
    print('left')

    return coreleft

def text_create(name, msg, directory):

    #Create a new TXT file and write the coordinate information

    full_path = directory + name + '.txt'
    file = open(full_path,'w')
    file.write(msg)
    file.close()
    print('Done')

if __name__ == '__main__':
    right = []
    left = []
    imagename = []
    directory = 'E:/OverMission/1086-190402/te-st/tes-t15/'
    # directory = 'E:/OverMission/1086-190402/only one test/'
    # directory = 'E:/1086-190402/test/'
    right, left, imagename = read_directory(directory, right, left, imagename)
    image = []

    i = 0
    for x in imagename:
        print('+++++++++++++++++++++++++++++++++++')
        print(x + '  begin')
        map =  cv2.imread(directory + x)
        cropright = map[int(right[i][1]):(int(right[i][1]) + int(right[i][3])),int(right[i][0]):(int(right[i][0]) + int(right[i][2]))]
        croplight = map[int(left[i][1]):(int(left[i][1]) + int(left[i][3])),int(left[i][0]):(int(left[i][0]) + int(left[i][2]))]
        coreright = []
        coreleft = []
        coreright = Right(cropright, right, coreright, map)
        coreleft = Left(croplight, left, coreleft, map)
        # text_create(x[:-4], 'r-spot: ' + str(coreright[0]) + '\n' + 'r-pupil: ' + str(coreright[1]) +'\n'+ 'l-spot: ' + str(coreleft[0]) +'\n'+ 'l-pupil: ' + str(coreleft[1]), directory)
        print(x + '  done!')

        i = i + 1
        image.append(cv2.imread(directory + x, 0))
