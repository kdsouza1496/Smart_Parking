centroids = []

centroids.append([1, 1])
centroids.append([9, 9])
centroids.append([5, 2])
centroids.append([2, 9])

temp1 = []

for c in centroids:
	if (c[:][1] < 8):
		temp1.append(c)

print temp1


'''
print centroids
temp1 = centroids[0:2]
temp2 = centroids[2:4]
print temp1
print temp2

centroids.sort(key=lambda x:x[1])
temp1 = list([centroids[i] for i in [0, 1]])
temp1.sort(key = lambda x:x[0])

temp2 = list([centroids[i] for i in [2, 3]])
temp2.sort(key = lambda x:x[0])

for t in temp2:
	temp1.append(t)

centroids = temp1

print centroids[3]
'''
