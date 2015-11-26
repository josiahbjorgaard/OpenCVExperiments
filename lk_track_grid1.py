#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Revised to calculate positions and velocities for total average velocity JAB 2015

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.screenx=700
        self.screeny=550

    def run(self):
        #Make the grid
        nptsx=10; nptsy=10;
        xgrid=np.linspace(0,self.screenx,nptsx)
        ygrid=np.linspace(0,self.screeny,nptsy)
        #Set up initial grid points for moving objects
        circlex,circley=np.meshgrid(xgrid,ygrid)
        circlex=np.reshape(circlex,-1)
        circley=np.reshape(circley,-1)
        self.p=np.reshape(zip(circlex,circley),[-1,1,2])
        circlevx=circlex*0.0;circlevy=circley*0.0;scale=0.1
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

	        #if there are points to track, track them
            if len(self.tracks) > 0:
                #Track current points

                img0, img1 = self.prev_gray, frame_gray #previous,current frames

                #Save old xy
                #print 'self tracks shape:',np.shape(self.tracks)
                old_tracks=self.tracks.reshape(-1,2) #save old tracks
                
                #track positions
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2) #latest points from current tracks recast for cv2
                #print 'p0 shape:',np.shape(p0.reshape(-1,2))
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) #track

                #Check if the detection is reversible
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) #track current backwords
                d = abs(p0-p0r).reshape(-1, 2).max(-1) #
                good = d < 1 #reversibility flag

                #Initialize variables
                new_tracks = []
                new_veloc = []
                N=0
                tot_veloc=0.0
                
                #Calculate velocities and store new coordinates
                for (xold, yold), (xnew, ynew) in zip(self.tracks.reshape(-1,2), p1.reshape(-1, 2)): #tracks,new po
                    #tr.append((x, y)) #append new points to track
                    ve=(xnew-xold,ynew-yold)
                    tr=(xnew,ynew) #Delta (x,y) instead of position 
                    new_tracks.append(tr) #add the updated track to the tracks (for deleting irreversible ones)
                    new_veloc.append(ve) #add the updated velocity
                #if len(new_veloc) > 1:
                #    self.veloc = new_veloc
                #    tot_veloc=np.mean(self.veloc,0)
                #else:
                #    tot_veloc=[0.0,0.0]                    
                self.tracks = np.reshape(new_tracks,[-1,1,2]) #update the track list with deleted irreversibles and updated points
                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0)) #draw lines
                #print np.shape(new_veloc),np.shape(circlex),np.shape(circlex)
                #UPDATE CIRCLE POSITION
                for cvx,cvy,cx,cy,(nvx,nvy) in zip(circlevx,circlevy,circlex,circley,new_veloc):
                    (cvx,cvy)=(cvx+nvx,cvy+nvy)
                    #(cx,cy)=(np.mod(cx+np.sign(cx)*cvx**2,self.screenx),np.mod(cy+np.sign(cy)*cvy**2,self.screeny))
                    (cx,cy)=(np.mod(cx+cvx,self.screenx),np.mod(cy+cvy,self.screeny))
                    cv2.circle(vis, (np.int32(cx), np.int32(cy)), 10, (0, 255, 0), 0)#-1)
                draw_str(vis, (20, 20), 'velocity x,y: %f %f, position x,y: %f %f' % (circlex[55], circley[55],circlevx[55],circlevy[55])) #write average velocity #NEWs
                #Check points to follow every detect_interval frames
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                #for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                #cv2.circle(mask, (x, y), 5, 0, -1)
                #p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                #If there are points from goodfeaturestotrack, make a matrix of (x,y)
                if self.p is not None:
                    #for x, y in np.float32(p).reshape(-1, 2): #switch result p to float and make list of x,y
                        #self.tracks.append([(x, y)])
                    self.tracks=np.float32(self.p) #shape is N,1,2
                    #print 'gFTT',np.shape(self.tracks)#

            self.frame_idx += 1 #count the frames
            self.prev_gray = frame_gray #switch to the next image
            cv2.imshow('lk_track', vis) #show the overlay of lines/circles

	    #Break key
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
