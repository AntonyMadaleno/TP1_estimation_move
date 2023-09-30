import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# Parameters for lucas kanade optical flow
lk_params = dict( 
                    winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

# params for ShiTomasi corner detection
feature_params = dict( 
                        maxCorners = 1000,
                        qualityLevel = 0.1,
                        minDistance = 10,
                        blockSize = 10 
                    )

def flow_LK(filepath, ext, k, lk_param, spacing, b = False):

    flow = None
    Ip = None
    P0 = None

    i = 1
    while True:

        In = None

        if i < 10 and b == True:
            In = cv2.imread(filepath + "0" + str(i) + "." + ext)
        else:
            In = cv2.imread(filepath + str(i) + "." + ext)

        if In is None:
            return flow.astype(np.uint8)

        if len(In.shape) == 3:
            In = cv2.cvtColor(In, cv2.COLOR_BGR2GRAY)

        if i == 1:
            Ip = In
            s = (Ip.shape[0] * Ip.shape[1]) // spacing
            array = np.arange(s) * spacing

            #p0 = cv2.goodFeaturesToTrack(Ip, mask = None, **feature_params)
            p0 = np.zeros( (Ip.shape[0] // spacing, Ip.shape[1] // spacing, 2), dtype= np.float32 )

            for n in range(0, p0.shape[0]):
                for m in range(0, p0.shape[1]):
                    p0[n,m,1] = n * spacing
                    p0[n,m,0] = m * spacing

            p0 = p0.reshape((p0.shape[0] * p0.shape[1], 1, 2))

            flow = np.zeros((In.shape[0], In.shape[1], 3, k), dtype = np.double)

        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(Ip, In, p0, None, **lk_params)

            good_new = None
            good_old = None

            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

            Ip = cv2.cvtColor(Ip, cv2.COLOR_GRAY2BGR)
            Ip = Ip*0

            # draw the tracks
            for x, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                dx = c-a
                dy = d-b
                L = (dx**2 + dy**2)**0.5

                tb = 0
                tr = 0

                if L > 4:
                    tb = abs( dx / L ) * 255.
                    tr = abs( dy / L ) * 255.
                    Ip = cv2.arrowedLine(Ip, (int(a), int(b)), (int(c), int(d)), ( tb, 255, tr), 1)
                #Ip = cv2.circle(Ip, (int(a), int(b)), 1, (200,0,0), -1)

            flow[:,:,:,i-1] = Ip
            Ip = In

        i = i+1

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
delt = frameDifferencing_movie("../toupie/toupie", "png", k)
flow = flow_LK("../toupie/toupie", "png", k, lk_params, 10)

h, w, c = flow[:,:,0].shape
fourcc = cv2.VideoWriter_fourcc(*'MP42') # FourCC is a 4-byte code used to specify the video codec.
video = cv2.VideoWriter('combined.mp4', fourcc, 10.0, (w, h))
 
for n in range(0,k-1):
    d = cv2.cvtColor(delt[:,:,n], cv2.COLOR_GRAY2BGR)
    video.write( cv2.add( d, flow[:,:,:,n] ).astype(np.uint8))
 
video.release()

p = 0
t = 1.0 / 10.0
while True:
    f = flow[:,:,:,p]
    d = delt[:,:,p]
    d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Video", cv2.add( f, d ) )
    p = (p+1)%(k-1)
    time.sleep(t)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' key is pressed