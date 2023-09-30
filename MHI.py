import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def func(m, x, r, t):

    if x > r:
        return t
    
    return max(0, m - 1)

def MHI(filepath, ext, t ,r = 30.0, b = False):

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
            FD = (FD / t) * 255.0
            return FD.astype(np.uint8)
        
        if len(In.shape) == 3:
            In = cv2.cvtColor(In, cv2.COLOR_BGR2GRAY)

        if i == 2:
            FD = np.zeros(In.shape, dtype= np.double)
        else:
            diff = abs(In.astype(np.double) - Ip.astype(np.double))
            FD = np.vectorize(func)(FD, diff, r, t)
        
        Ip = In

##Q2##

pirate = MHI("../pirate/frame", "png", 12)
course = MHI("../course/frame", "png", 40)
femme = MHI("../femme/frame", "png", 22)
action = MHI("../action/seq2_0", "png", 25, b = True)

cv2.imwrite("../.results/images/pirate_MHI.png", pirate)
cv2.imwrite("../.results/images/course_MHI.png", course)
cv2.imwrite("../.results/images/femme_MHI.png", femme)
cv2.imwrite("../.results/images/action_MHI.png", action)

fig = plt.figure(figsize = (12,7))
columns = 2
rows = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(pirate, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('pirate_MHI')

fig.add_subplot(rows, columns, 2)
plt.imshow(course, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('course_MHI')

fig.add_subplot(rows, columns, 3)
plt.imshow(femme, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('femme_MHI')

fig.add_subplot(rows, columns, 4)
plt.imshow(action, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('action_MHI')

plt.show()