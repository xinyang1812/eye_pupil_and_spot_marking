import os
import cv2
'''
This code is used to detect the correctness of the selected eye area, pupil and spot coordinates(by chendw).     19.06.03
'''

def splitline(spot):

    # Segmentation of coordinate points saved in text
    spot = spot.split(',')
    ss = len(spot) - 1
    _, spot[0] = spot[0].split('[')
    spot[ss], _ = spot[ss].split(']')
    spots = []
    if len(spot) == 4:
        spots.append((int(spot[0]), int(spot[1])))
        spots.append((int(spot[2]), int(spot[3])))
    else:
        spots.append((int(spot[0]), int(spot[1])))
    return spots

def main():
    rootdir = 'E:/OverMission/1086-190402/te-st/t-15/'
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isfile(path):
               if path[-4:] == '.bmp':
                  path[:-4]
                  f = open(path[:-4]+'.txt')
                  txt = f.read()
                  re = txt.split('\n')

                  # Extraction of coordinate points in text
                  _, rspot = re[0].split(':')
                  rspots = splitline(rspot)
                  _, rpupil = re[1].split(':')
                  rpupils = splitline(rpupil)
                  _, lspot = re[2].split(':')
                  lspots = splitline(lspot)
                  _, lpupil = re[3].split(':')
                  lpupils = splitline(lpupil)

                  img = cv2.imread(path)
                  r0, r1, r2, r3, r4, r5 = path[:-4].split('-')
                  rx, ry, rh,rw = r4[1:].split(',')
                  lx, ly, lh, lw = r5[1:].split(',')

                  #The color, size, form of the dots drawn
                  point_size = 1
                  point_color = (0, 0, 255)  # BGR
                  thickness = 1

                  # Draw coordinate points on a graph
                  for point in rspots:
                        cv2.circle(img, point, point_size, point_color, thickness)
                  for point in rpupils:
                        cv2.circle(img, point, point_size, point_color, thickness)
                  for point in lspots:
                        cv2.circle(img, point, point_size, point_color, thickness)
                  for point in lpupils:
                        cv2.circle(img, point, point_size, point_color, thickness)

                  # Draw the frame on the picture.
                  cv2.rectangle(img, (int(rx), int(ry)), (int(rx)+int(rh),int(ry)+int(rw)), (0, 255, 0), 1)
                  cv2.rectangle(img, (int(lx), int(ly)), (int(lx) + int(lh), int(ly) + int(lw)), (0, 255, 0), 1)

                  imr = img[int(ry):(int(ry)+int(rw)),int(rx):(int(rx)+int(rh))]
                  w, h = imr.shape[0], imr.shape[1]
                  imr = cv2.resize(imr, (h * 10, w * 10), interpolation=cv2.INTER_CUBIC)

                  iml = img[int(ly):(int(ly) + int(lw)),int(lx):(int(lx) + int(lh))]
                  w, h = iml.shape[0], iml.shape[1]
                  iml = cv2.resize(iml, (h * 10, w * 10), interpolation=cv2.INTER_CUBIC)

                  # cv2.imwrite('test.jpg', img)
                  print (path)
                  cv2.imshow('right',imr)
                  cv2.waitKey(100)

                  cv2.imshow('left',iml)
                  cv2.waitKey(100)


if __name__ == '__main__':
    main()

