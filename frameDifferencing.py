import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def frameDifferencing(filepath, ext, b = False):

    Ip = None
    FD = None
    i = 1
    while True:

        In = None

        if i < 10 and b == True:
            In = cv2.imread(filepath + "0" + str(i) + "." + ext)
        else:
            In = cv2.imread(filepath + str(i) + "." + ext)

        i += 1

        if In is None:
            M = np.max(FD)
            FD = (FD / M) * 255.0
            return FD.astype(np.uint8)
        
        if len(In.shape) == 3:
            In = cv2.cvtColor(In, cv2.COLOR_BGR2GRAY)

        if i == 2:
            FD = np.zeros(In.shape, dtype= np.double)
        else:
            FD = FD + abs(In.astype(np.double) - Ip.astype(np.double))
        
        Ip = In

##Q2##

neige = frameDifferencing("../neige/frame", "png")
course = frameDifferencing("../course/frame", "png")
femme = frameDifferencing("../femme/frame", "png")
action = frameDifferencing("../action/seq2_0", "png", b = True)

cv2.imwrite("../.results/images/neige_frameDiff.png", neige)
cv2.imwrite("../.results/images/course_frameDiff.png", course)
cv2.imwrite("../.results/images/femme_frameDiff.png", femme)
cv2.imwrite("../.results/images/action_frameDiff.png", action)

fig = plt.figure(figsize = (12,7))
columns = 2
rows = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(neige, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('neige_frameDiff')

fig.add_subplot(rows, columns, 2)
plt.imshow(course, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('course_frameDiff')

fig.add_subplot(rows, columns, 3)
plt.imshow(femme, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('femme_frameDiff')

fig.add_subplot(rows, columns, 4)
plt.imshow(action, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('action_frameDiff')

plt.show()
