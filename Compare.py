import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

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


######COMPARISON#####
paths = ["../homme/frame", "../diplome/frame", "../lumiere/frame", "../toupie/toupie"]
k = [18, 31, 17, 10]

for i in range (0, 4):
    mhi = MHI(paths[i], "png", k[i])
    avg = computeAverageBackground(paths[i], "png")
    avg = compareToFrames(avg, paths[i], "png")
    fdiff = frameDifferencing(paths[i], "png")

    fig = plt.figure(figsize = (12,7))
    columns = 3
    rows = 1

    fig.add_subplot(rows, columns, 1)
    plt.imshow(mhi, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.title('MHI')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(avg, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.title('AVG')

    fig.add_subplot(rows, columns, 3)
    plt.imshow(fdiff, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.title('fdiff')

    plt.show()
