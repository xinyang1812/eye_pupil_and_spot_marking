# -*- coding: UTF-8 -*-
import numpy as np
import os
import cv2
import imutils

def read_directory(directory_name, right, light, imagename):
    R = [[]for i in range(1)]
    L = [[]for i in range(1)]
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
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    index = [[] for i in range(1)]
    # loop over the contours
    count = 0
    for c in cnts:
        if count < 2:
            if len(c) > 3:
                # compute the center of the contour
                try:
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print (cX, cY)
                    count = count + 1
                    index[0].append(cX)
                    index[0].append(cY)
                    # cv2.namedWindow('core')
                    cv2.circle(img_gray, (cX, cY), 1, (0, 0, 213), -1)
                    # cv2.imshow('core', img_gray)
                    # cv2.waitKey(0)
                except:
                    continue
    return index[0]

def Right(cropright, right, coreright, map):
    img_gray_cropright = cv2.cvtColor(cropright, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(img_gray_cropright, 160, 255, cv2.cv2.THRESH_BINARY)


    # cv2.imshow('dsds', thresh1)
    # cv2.waitKey(0)
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
    ret, thresh2 = cv2.threshold(dst, 225, 255, cv2.cv2.THRESH_BINARY)

    index = []
    # cv2.imshow('ds', thresh2)
    # cv2.waitKey(0)
    index = core(thresh2, cropright)

    index[0] = index[0] + int(right[i][0])
    index[1] = index[1] + int(right[i][1])
    coreright.append(index)

    # cv2.namedWindow('ds')
    # # cv2.circle(map, (coreright[0][0], coreright[0][1]), 1, (0, 0, 213), -1)
    # # cv2.circle(map, (coreright[0][2], coreright[0][3]), 1, (0, 0, 213), -1)
    # # cv2.circle(map, (coreright[1][0], coreright[1][1]), 1, (0, 0, 213), -1)
    # cv2.imshow('ds', map)
    # cv2.waitKey(100)
    print('right')

    return coreright





def Left(cropleft, left, coreleft, map):
    img_gray_cropleft = cv2.cvtColor(cropleft, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(img_gray_cropleft, 160, 255, cv2.cv2.THRESH_BINARY)
    # cv2.imshow('dsds', thresh1)
    # cv2.waitKey(0)
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

    ret, thresh2 = cv2.threshold(dst, 230, 255, cv2.cv2.THRESH_BINARY)
    # cv2.imshow('dsds', thresh2)
    # cv2.waitKey(0)

    index = []
    index = core(thresh2, cropleft)
    index[0] = index[0] + int(left[i][0])
    index[1] = index[1] + int(left[i][1])
    coreleft.append(index)

    cv2.namedWindow('ds')
    cv2.imshow('ds', map)
    cv2.waitKey(100)
    print('left')

    return coreleft

    # cv2.namedWindow('ds')
    # # cv2.circle(map, (index[0], index[1]), 1, (0, 0, 213), -1)
    # # cv2.circle(map, (index[2], index[3]), 1, (0, 0, 213), -1)
    # cv2.imshow('ds', map)
    # cv2.waitKey(0)





if __name__ == '__main__':
    right = []
    left = []
    imagename = []
    directory = 'E:/1086-190402/9-pointsnew/'
    # directory = 'E:/1086-190402/test/'
    right, left, imagename = read_directory(directory, right, left, imagename)
    image = []
    coreright = []
    coreleft = []
    i = 0
    for x in imagename:
        print('+++++++++++++++++++++++++++++++++++')
        print(x + '  begin')
        map =  cv2.imread(directory + x)
        cropright = map[int(right[i][1]):(int(right[i][1]) + int(right[i][3])),int(right[i][0]):(int(right[i][0]) + int(right[i][2]))]
        croplight = map[int(left[i][1]):(int(left[i][1]) + int(left[i][3])),int(left[i][0]):(int(left[i][0]) + int(left[i][2]))]
        coreright = Right(cropright, right, coreright, map)
        coreleft = Left(croplight, left, coreleft, map)
        print(x + '  done!')

        # cv2.namedWindow('crop')
        # cv2.imshow('crop', thresh1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        i = i + 1
        image.append(cv2.imread(directory + x, 0))
    print('sd')
