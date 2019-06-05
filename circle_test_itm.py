# -*- coding: UTF-8 -*-
import numpy as np
import os
import cv2
import imutils
from PIL import Image
from pylab import *

"""
This edition is the second edition used to locate the eye center coordinates and spot coordinates. 
It mainly uses manual getinput to mark the spot coordinates, and uses Hough circle detection to 
determine the pupil coordinates.   19.06.04
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


def cicle(gray):
    # 霍夫变换圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=10, minRadius=1, maxRadius=15)

    for circle in circles[0]:
        print(circle[2])
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        gray = cv2.circle(gray, (x, y), r, (255, 0, 0), 1, 8, 0)
        w, h = gray.shape
        cv2.circle(gray, (x, y), 1, (255, 0, 0), -1)
        # gray = cv2.resize(gray, (h*4, w*4), interpolation=cv2.INTER_CUBIC)
        #
        # cv2.imshow('5', gray)
        # cv2.waitKey(100)
    return x,y,r

def Right(cropright, right, coreright, map, imagename, directory):

    # Processing the right eye and returning to the pupil center and spot coordinates of the right eye

    img_gray_cropright = cv2.cvtColor(cropright, cv2.COLOR_RGB2GRAY)

    # Finding the centroid coordinates of the pupil（single）
    ii = 0
    index = [[] for ii in range(2)]
    x, y, r = cicle(img_gray_cropright)
    index[0] = int(x) + int(right[i][0])
    index[1] = int(y) + int(right[i][1])
    coreright.append(index)
    # cv2.circle(map, (index[0], index[1]), 1, (0, 0, 213), -1)

    # Finding the centroid coordinates of the spot（Multiple）
    image = Image.fromarray(cv2.cvtColor(cropright, cv2.COLOR_BGR2RGB))
    imshow(image)
    xxx = ginput(2)
    plt.close()
    index = [[] for ii in range(4)]
    index[0] = int(xxx[0][0]) + int(right[i][0])
    index[1] = int(xxx[0][1]) + int(right[i][1])
    index[2] = int(xxx[1][0]) + int(right[i][0])
    index[3] = int(xxx[1][1]) + int(right[i][1])
    coreright.append(index)
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
    coreleft.append(index)
    # cv2.circle(map, (index[0], index[1]), 1, (0, 0, 213), -1)

    # Finding the centroid coordinates of the spot（Multiple）
    image = Image.fromarray(cv2.cvtColor(cropleft, cv2.COLOR_BGR2RGB))
    imshow(image)
    xxx = ginput(2)
    plt.close()
    index = [[] for ii in range(4)]
    index[0] = int(xxx[0][0]) + int(left[i][0])
    index[1] = int(xxx[0][1]) + int(left[i][1])
    index[2] = int(xxx[1][0]) + int(left[i][0])
    index[3] = int(xxx[1][1]) + int(left[i][1])
    coreleft.append(index)
    return coreleft

def text_create(name, msg, directory):

    #Create a new TXT file and write the coordinate information

    full_path = directory + name + '.txt'
    file = open(full_path,'w')
    file.write(msg)
    file.close()
    print('Done')



if __name__=='__main__':
    right = []
    left = []
    imagename = []
    directory = 'E:/OverMission/1086-190402/te-st/tes-t9/5/'
    # directory = 'E:/OverMission/1086-190402/only one test/'
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

        text_create(x[:-4], 'r-spot: ' + str(coreright[1]) + '\n' + 'r-pupil: ' + str(coreright[0]) +'\n'+ 'l-spot: ' + str(coreleft[1]) +'\n'+ 'l-pupil: ' + str(coreleft[0]), directory)
        print(x + '  done!')

        i = i + 1
        image.append(cv2.imread(directory + x, 0))
