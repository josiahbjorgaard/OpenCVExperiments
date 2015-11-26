#!/usr/bin/env python

'''
TODO: save read video + new capture to new video and read new video as read video every N timesteps
FIXME: Time counter restarts
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
        self.switch_time=10
        self.codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
        self.total_writer=cv2.VideoWriter("total"+".wmv", self.codec, 10, (640,480), 1)

        cv2.namedWindow("Image")
        #cv2.createTrackbar("Detection treshold: ", "Image", self.threshold, 100)
        
    def run(self):
        
        fgbg = cv2.createBackgroundSubtractorMOG2(500,16,False)#,detectShadows=False)  

        #Initial variables
        start_time=clock()
        time_counter=0
        restart=False
        switch=True
        initialize=True
        N=0
        #Main loop
        while True:
            #Read from camera, background subtraction     
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(frame_gray)
            newframe = cv2.bitwise_and(frame,frame,mask = fgmask) #apply mask to frame  
            
            #Reading from file, restart and switch between two files to layer images
            if time_counter == 0: #Initial step
                print 'Time 0'
                if switch is True: #switch the files (current capture, all captures) every time around
                    switch=False
                    self.writer=cv2.VideoWriter("save1"+".wmv", self.codec, 10, (640,480), 1)
                else:
                    switch=True
                    self.writer=cv2.VideoWriter("save2"+".wmv", self.codec, 10, (640,480), 1)
            elif time_counter < self.switch_time: #read all captures, write with added current capture
                #print 'good time'
                if initialize is False: #first initialize (must have the first video saved)
                    ret,saved_frame=self.saved_video.read()
                    #print 'size',np.shape(saved_frame),np.shape(newframe)
                    #print type(saved_frame)
                    if (type(saved_frame) != type(None)):
                        newframe=cv2.add(saved_frame,newframe)
                        frame=cv2.add(frame,saved_frame)
                self.writer.write(newframe) #save newframe (initial or added) to file
            elif time_counter > self.switch_time:
                #print 'over time'
                restart=True

                if initialize is True:
                    initialize = False
                else:
                    if (type(saved_frame) == type(None)):
                        restart=True
                    self.saved_video.release()
                if self.writer.isOpened() is True:
                    print 'Release writer'
                    self.writer.release()
                if switch is True:
                    print('Video switch is true')
                    self.saved_video=video.create_capture('save2.wmv')
                else:
                    print('video switch is false')
                    self.saved_video=video.create_capture('save1.wmv')
                print "INITIALIZED VIDEO"


            #Display images            
            draw_str(frame, (20, 20), 'Time counter: %f' % time_counter)
            cv2.imshow('Image',frame)            
            cv2.imshow('Frame',newframe)
            self.total_writer.write(frame) #save newframe (initial or added) to file

            #Time check and update
            time_counter=clock()-start_time
            #print time_counter
            if restart is True:
                print 'RESTART'
                N=N+1
                print 'N',N
                restart=False
                time_counter=0.0
                start_time=clock()
                
            #Break the program    
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
                
        self.cam.release()
        self.writer.release()
        self.total_writer.release()
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