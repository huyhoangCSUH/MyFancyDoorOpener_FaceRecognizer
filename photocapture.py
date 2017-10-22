from cv2 import *
import sys

cam = VideoCapture(0)
numOfPhoto = int(sys.argv[1])
for i in range(0,numOfPhoto):
	s, img = cam.read()
	if s:
		imwrite("/Users/huyhoang/Desktop/" + str(i) + ".jpg",img)
