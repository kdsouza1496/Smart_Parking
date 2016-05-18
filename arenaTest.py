import numpy as np
import cv2

#area of the bounding box square corners.
refThresh = 200

#Size of the Region of interest. (within the bounding box)
arenaWidth = 800
arenaHeight = 600

cap = cv2.VideoCapture("./v1.avi")
car_cascade = cv2.CascadeClassifier('cascade.xml')


def getArena(frame):
	#rotate the image clockwise to induce the tilt error in only one direction
	M = cv2.getRotationMatrix2D((width, height),-3,1)
	frame = cv2.warpAffine(frame,M,(width,height))
		
	#print "%d %d" %(height, width)
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_green= np.array([50,50,50])
	upper_green = np.array([70,255,255])
	
	mask = cv2.inRange(hsv, lower_green, upper_green)
	resGreen = cv2.bitwise_and(frame,frame, mask= mask)
	
	gray = cv2.cvtColor(resGreen, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	i=0
	j=0
	
	centroids = np.empty([4, 2], dtype=int)
	
	while(i < len(contours)):
		area = cv2.contourArea(contours[i])
		if(area > refThresh):
			#print area
			cv2.drawContours(frame, contours, i, (127,127,127), 1)
			M = cv2.moments(contours[i])
			centroids[j][0] = int(M['m10']/M['m00'])
			centroids[j][1] = int(M['m01']/M['m00'])
			j=j+1
			
		i += 1
	
	k = j
	
	#print centroids
	#j -= 1
	#while(j >= 0):
	#	print "%d %d" %(centroids[j][0], centroids[j][1])
	#	j -= 1
	
	if(k == 4):
		pts1 = np.float32([centroids[3], centroids[2], centroids[0], centroids[1]])
		#print pts1
		pts2 = np.float32([[0, 0], [arenaWidth, 0], [arenaWidth, arenaHeight], [0, arenaHeight]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		arena = cv2.warpPerspective(frame,M,(arenaWidth,arenaHeight))
	
		arena = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
	
		return arena
	
	else:
		print j 
		return []

sampleWidth = 160
sampleHeight = 100
count = 0
	
while(cap.isOpened()):
	ret, frame = cap.read()
	original = frame
	
	height, width, channels = frame.shape
	
	arena = getArena(frame)
	
	if(len(arena) == 0):
		print "Could not crop"
		continue

	toDisplay = arena
	toDisplay = cv2.resize(toDisplay,(800, 600), interpolation = cv2.INTER_CUBIC)
	cv2.imshow('Cropped',toDisplay)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
	
	
		 


		
