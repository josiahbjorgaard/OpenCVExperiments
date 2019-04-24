#!/usr/bin/env python

'''
Move the ball with the webcam

Usage
-----
Run the python file and move infront of the webcam

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
        circlex=100;circley=100;circlevx=0.0;circlevy=0.0;scale=0.1
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

	    #if there are points to track, track them
            if len(self.tracks) > 0:
                #Track current points
                img0, img1 = self.prev_gray, frame_gray #previous,current frames

                #Save old xy
                old_tracks=self.tracks.reshape(-1,2) #save old tracks
                
                #track positions
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2) #latest points from current tracks recast for cv2
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
                for (xold, yold), (xnew, ynew), good_flag in zip(self.tracks.reshape(-1,2), p1.reshape(-1, 2), good): #tracks,new points, reversibility flag
                    if not good_flag: #skip the point (delete the track) if it's not reversible
                        continue
                    #tr.append((x, y)) #append new points to track
                    ve=(xnew-xold,ynew-yold)
                    tr=(xnew,ynew) #Delta (x,y) instead of position 

                    if len(tr) > self.track_len: #if the track is longer than the set limit, eliminate the oldest point
                        del tr[0]

                    new_tracks.append(tr) #add the updated track to the tracks (for deleting irreversible ones)
                    new_veloc.append(ve) #add the updated velocity
                if len(new_veloc) > 1:
                    self.veloc = new_veloc
                    tot_veloc=np.mean(self.veloc,0)
                else:
                    print 'Zeroing'
                    tot_veloc=np.array([0.0,0.0])
                    
                self.tracks = np.reshape(new_tracks,[-1,1,2]) #update the track list with deleted irreversibles and updated points
                draw_str(vis, (20, 20), 'velocity x,y: %f %f, position x,y: %f %f' % (tot_veloc[0], tot_veloc[1],np.int32(circlex),np.int32(circley))) #write average velocity #NEWs

                #UPDATE CIRCLE POSITION
                if tot_veloc.any() < 50: #FIXME
                    circlevx+=tot_veloc[0]*scale
                    circlevy+=tot_veloc[1]*scale
                circlex+=np.sign(circlevx)*circlevx**2
                circley+=np.sign(circlevy)*circlevy**2
                circlex=np.mod(circlex,self.screenx)
                circley=np.mod(circley,self.screeny)
                cv2.circle(vis, (np.int32(circlex), np.int32(circley)), 20, (0, 255, 0), -1)#-1)
                #Check points to follow every detect_interval frames
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                #If there are points from goodfeaturestotrack, make a matrix of (x,y)
                if p is not None:
                    self.tracks=np.float32(p) #shape is N,1,2

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
