import numpy as np
import cv2
import urllib

#area of the bounding box square corners.
refThresh = 10

#Size of the Region of interest. (within the bounding box)
arenaWidth = 800
arenaHeight = 600

car_cascade = cv2.CascadeClassifier('12.xml')

def swap(a, x, y):
	temp = a[x]
	a[x] = a[y]
	a[y] = temp
	return a

#cap = cv2.VideoCapture("./v1.avi")

def url_to_image(url):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype = "uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

def getArena(frame):
	height, width, channels = frame.shape

	#rotate the image clockwise to induce the tilt error in only one direction
	M = cv2.getRotationMatrix2D((width, height),-5,1)
	frame = cv2.warpAffine(frame,M,(width,height))
	

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_green= np.array([50,50,50])
	upper_green = np.array([70,255,255])
	
	mask = cv2.inRange(hsv, lower_green, upper_green)
	#resGreen = cv2.bitwise_and(frame,frame, mask= mask)
	
	#gray = cv2.cvtColor(resGreen, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	#cv2.drawContours(mask, contours, -1, (255, 255, 255), 3)

	areas = []
	index = []
	
	i = 0

	while(i < len(contours)):
		area = cv2.contourArea(contours[i])
		if(area > refThresh):
			areas.append(area)
			index.append(i)				
		i += 1	
	
	areas = np.array(areas)

	index = np.array(index)

	#zipped = zip(areas, index)
	#zipped.sort(reverse=True)
	
	#index = [ind for area, ind in zipped]

	corners = index

	centroids = []

	for corner in corners:	
		cv2.drawContours(frame, contours, corner, (127,127,127), 3)
		M = cv2.moments(contours[corner])
		
		centroid = []
		centroid.append(int(M['m10']/M['m00']))
		centroid.append(int(M['m01']/M['m00']))
	
		centroids.append(centroid)
	
	k = len(centroids)
	print k

	#sort them according to y axis
	centroids.sort(key=lambda x:x[1])

	temp1 = []

	#Get all centroids in the upper half
	for c in centroids:
		if(c[:][1] < height/2):
			temp1.append(c)
	
	#Sort them according to x axis
	temp1.sort(key = lambda x:x[0])
	

	temp2 = []

	#Get all centroids in the lower half
	for c in centroids:
		if(c[:][1] > height/2):
			temp2.append(c)

	#Sort them according to x axis
	temp2.sort(key = lambda x:x[0])

	centroids = []

	#Left most top centroid
	centroids.append(temp1[0])
	#Right most top centroid
	centroids.append(temp1[len(temp1) - 1])
	
	#Left most bottom centroid
	centroids.append(temp2[0])

	#Right most bottom centroid
	centroids.append(temp2[len(temp2) - 1])

	print centroids
	
	#Adjust the perception distortion.

	#Initial points
	pts1 = np.float32([centroids[0], centroids[1], centroids[2], centroids[3]])
	
	#Points to map in the final image
	pts2 = np.float32([[0, 0], [arenaWidth, 0], [0, arenaHeight], [arenaWidth, arenaHeight]])
	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	M = cv2.getPerspectiveTransform(pts1,pts2)
	arena = cv2.warpPerspective(frame,M,(arenaWidth,arenaHeight))
	
	return arena
	

sampleWidth = 160
sampleHeight = 100
count = 0
indices = []

r0 = range(20, 150)
r1 = range(250, 380)
r2 = range(420, 550)
r3 = range(650, 780)

range_y = range(20, 80) + range(120, 180) + range(420, 480) + range(520, 580)
range_x = range(20, 150) + range(250, 380) + range(420, 550) + range(650, 780)

def getIndex(x1, y1):
	y2 = y1/100
	if(y2 >= 4):
		y2 -= 2

	if(x1 in r0):
		x2 = 0

	elif(x1 in r1):
		x2 = 1

	elif(x1 in r2):
		x2 = 2

	elif(x1 in r3):
		x2 = 3

	return [x2, y2] 


while(1):
	frame = url_to_image('http://192.168.43.1:8080/shot.jpg')
	indices = []	
	
	#ret, frame = cap.read()
	original = frame
	
	height, width, channels = frame.shape
	print width
	print height

	toDisplay = frame
	toDisplay = cv2.resize(toDisplay,(800, 600), interpolation = cv2.INTER_CUBIC)
	cv2.imshow('Cropped',toDisplay)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	continue	

	arena = getArena(frame)
	
	if(len(arena) == 0):
		print "Could not crop"
		continue

	cars = car_cascade.detectMultiScale(arena, 1.5, 15)
	
	for (x,y,w,h) in cars:
		center_x = x + w/2
		center_y = y + h/2

		if( (center_x in range_x) and (center_y in range_y) ):
			index = getIndex(center_x, center_y)
			if(index not in indices):
				cv2.rectangle(arena,(x,y),(x+w,y+h),(255,0,0),2)
				print index
				indices.append(index)
	
	toDisplay = arena
	toDisplay = cv2.resize(toDisplay,(800, 600), interpolation = cv2.INTER_CUBIC)
	cv2.imshow('Cropped',toDisplay)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
	
	
		 


		
