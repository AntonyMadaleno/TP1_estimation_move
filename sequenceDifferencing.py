import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def frameDifferencing_movie(filepath, ext, k):

    Ip = cv2.imread(filepath + str(1) + "." + ext)
    Ip = cv2.cvtColor(Ip, cv2.COLOR_BGR2GRAY)
    Deltas = np.zeros( (Ip.shape[0], Ip.shape[1], k-1), dtype= np.uint8 )

    for i in range(1, k):

        In = cv2.imread(filepath + str( i + 1 ) + "." + ext)
        In = cv2.cvtColor(In, cv2.COLOR_BGR2GRAY)

        if In is None or Ip is None:
            return
        
        Diff = np.abs(In.astype(np.double) - Ip.astype(np.double))
        Deltas[:,:,i-1] = Diff
        Ip = In

    return Deltas.astype(np.uint8)

k = 10
flow = frameDifferencing_movie("../toupie/toupie", "png", k)

h, w = flow[:,:,0].shape
fourcc = cv2.VideoWriter_fourcc(*'MP42') # FourCC is a 4-byte code used to specify the video codec.
video = cv2.VideoWriter('sequenceDifferencing.mp4', fourcc, 10.0, (w, h))
 
for n in range(0,k-1):
    video.write(flow[:,:,n].astype(np.uint8))
 
video.release()

p = 0
t = 1.0 / 10.0
while True:
    cv2.imshow("Video", flow[:,:,p])
    p = (p+1)%(k-1)
    time.sleep(t)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' key is pressed