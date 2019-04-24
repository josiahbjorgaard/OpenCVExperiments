#!/usr/bin/env python

'''
TODO: save read video + new capture to new video and read new video as read video every N timesteps
FIXME: sometimes it goes over switch time
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
        self.cam = video.create_capture(video_src)        
        self.screenx=700
        self.screeny=550
        self.threshold=50
        self.switch_time=3
        self.codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')

        cv2.namedWindow("Image")
        #cv2.createTrackbar("Detection treshold: ", "Image", self.threshold, 100)
        
    def run(self):
        
        fgbg = cv2.createBackgroundSubtractorMOG2(history=10000)  
        #ret, frame0 = self.cam.read()
        start_time=clock()
        time_counter=0
        restart=False
        while True:     
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(frame_gray)
            newframe = cv2.bitwise_and(frame,frame,mask = fgmask)  
            print time_counter
            if time_counter == 0:
                self.writer1=cv2.VideoWriter("save1"+".wmv", self.codec, 10, (640,480), 1)
                self.writer2=cv2.VideoWriter("save2"+".wmv", self.codec, 10, (640,480), 1)
            elif time_counter < self.switch_time:
                self.writer1.write(newframe)
            elif time_counter > self.switch_time:
                if self.writer1.isOpened() is True:
                    self.writer1.release()
                    self.video1=video.create_capture('save1.wmv')
                    print "INITIALIZING VIDEO"
                else:
                    red,saved_frame=self.video1.read()
                    if (type(saved_frame) == type(None)):
                        restart=True
                    else:
                        frame=cv2.add(saved_frame,frame)
            draw_str(frame, (20, 20), 'Time counter: %f' % time_counter)
            cv2.imshow('Image',frame)            
            cv2.imshow('Frame',newframe)
            #cv2.imshow('frame',frame)
            time_counter=clock()-start_time

            if restart is True:
                restart=False
                time_counter=0.0
                start_time=clock()
                
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break    
        self.cam.release()
        
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