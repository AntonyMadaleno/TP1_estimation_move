import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def computeAverageBackground(filepath, ext, b = False):

    sumImage = None

    i = 1
    while True:

        image = None

        if i < 10 and b == True:
            image = cv2.imread(filepath + "0" + str(i) + "." + ext)
        else:
            image = cv2.imread(filepath + str(i) + "." + ext)

        i = i+1

        if image is None:
            sumImage = sumImage / (i-1)
            return sumImage.astype(np.uint8)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if i == 2:
            sumImage = np.zeros(image.shape, dtype = np.double)

        sumImage = sumImage + image.astype(np.double)

def compareToFrames(I, filepath, ext, b = False):
    
    R = np.zeros(I.shape, dtype= np.double)

    i = 1
    while True:

        if i < 10 and b == True:
            filepath = filepath + "0"

        image = cv2.imread(filepath + str(i) + "." + ext)
        i = i+1

        if image is None:
            M = np.max(R)
            R = (R / M) * 255.0
            return R.astype(np.uint8)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        R = R + np.abs(image.astype(np.double) - I.astype(np.double))

        
####Q1####
neige = computeAverageBackground("../neige/frame", "png")
course = computeAverageBackground("../course/frame", "png")
pirate = computeAverageBackground("../pirate/frame", "png")
action = computeAverageBackground("../action/seq2_0", "png", b=True)

cv2.imwrite("../.results/images/neige_avgBackground.png", neige)
cv2.imwrite("../.results/images/course_avgBackground.png", course)
cv2.imwrite("../.results/images/pirate_avgBackground.png", pirate)
cv2.imwrite("../.results/images/action_avgBackground.png", action)

r_neige = compareToFrames(neige, "../neige/frame", "png")
r_course = compareToFrames(course, "../course/frame", "png")
r_pirate = compareToFrames(pirate, "../pirate/frame", "png")
r_action = compareToFrames(action, "../action/seq2_0", "png", b=True)

cv2.imwrite("../.results/images/neige_Q1.png", r_neige)
cv2.imwrite("../.results/images/course_Q1.png", r_course)
cv2.imwrite("../.results/images/pirate_Q1.png", r_pirate)
cv2.imwrite("../.results/images/action_Q1.png", r_action)

fig = plt.figure(figsize = (12,7))
columns = 2
rows = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(neige, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('neige_avgBackground')

fig.add_subplot(rows, columns, 2)
plt.imshow(course, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('course_avgBackground')

fig.add_subplot(rows, columns, 3)
plt.imshow(pirate, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('pirate_avgBackground')

fig.add_subplot(rows, columns, 4)
plt.imshow(action, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('action_avgBackground')

plt.show()

fig = plt.figure(figsize = (12,7))
columns = 2
rows = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(r_neige, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('neige_q1')

fig.add_subplot(rows, columns, 2)
plt.imshow(r_course, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('course_q1')

fig.add_subplot(rows, columns, 3)
plt.imshow(r_pirate, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('pirate_q1')

fig.add_subplot(rows, columns, 4)
plt.imshow(r_action, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('action_q1')

plt.show()