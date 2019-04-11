#!/usr/bin/env python

import cv2 as cv
import numpy as np
from datetime import datetime
import time

class MotionDetectorInstantaneous():
    
    def onChange(self, val): #callback when the user change the detection threshold
        self.threshold = val
    
    def __init__(self,threshold=8, doRecord=True, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord=doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        self.frame = None
    
        self.capture=cv.VideoCapture(0)
        result, self.frame = self.capture.read() #Take a frame to init recorder
        if doRecord:
            self.initRecorder()
        self.height, self.width, self.channels = self.frame.shape
        self.frame1gray = np.zeros((self.height, self.width, 1), np.uint8) #cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t-1
        self.frame1gray=cv.cvtColor(self.frame, cv.COLOR_RGB2GRAY)
 
        #Will hold the thresholded result
        self.res = np.zeros((self.height, self.width, 3), np.uint8) #cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U)
        
        self.frame2gray = np.zeros((self.height, self.width, 1), np.uint8) #cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t
        
        self.nb_pixels = self.width * self.height
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 #Hold timestamp of the last detection
        
        if showWindows:
            cv.namedWindow("Image")
            cv.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)
        
    def initRecorder(self): #Create the recorder
        codec = cv.VideoWriter_fourcc('W', 'M', 'V', '2')
        self.height, self.width, _ = self.frame.shape
        self.writer=cv.VideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".mpg", codec, 5, (self.height,self.width), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.FONT_HERSHEY_SIMPLEX
        #self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font

    def run(self):
        started = time.time()
        while True:
            
            result, curframe = self.capture.read()
            instant = time.time() #Get timestamp o the frame
            
            self.processImage(curframe) #Process the image
            
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant #Update the trigger_time
                    if instant > started +5:#Wait 5 second after the webcam start for luminosity adjusting etc..
                        print([datetime.now().strftime("%b %d, %H:%M:%S"), "Something is moving !"])
                        if self.doRecord: #set isRecording=True only if we record a video
                            self.isRecording = True
            else:
                if instant >= self.trigger_time +10: #Record during 10 seconds
                    print([datetime.now().strftime("%b %d, %H:%M:%S"), "Stop recording"])
                    self.isRecording = False
                else:
                    cv.putText(curframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 1, (255,255,255),2) #Put date on the frame
                    self.writer.write(curframe) #Write the frame
            
            if self.show:
                cv.imshow("Image", curframe)
                cv.imshow("Res", self.res)
                
            #cv.copy(self.frame2gray, self.frame1gray)
            self.frame1gray=self.frame2gray.copy()
            c=cv.waitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break            
    
    def processImage(self, frame):
        self.frame2gray=cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        #Absdiff to get the difference between to the frames
        self.res = np.zeros((self.height, self.width, 1), np.uint8)
        cv.absdiff(self.frame1gray, self.frame2gray, self.res)

        #Remove the noise and do the threshold
        #cv.Smooth(self.res, self.res, cv.CV_BLUR, 5,5)
        self.res=cv.morphologyEx(self.res, cv.MORPH_OPEN, None)#self.res)
        #cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_OPEN)
        self.res=cv.morphologyEx(self.res, cv.MORPH_CLOSE, None)#self.res)
        #cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_CLOSE)
        ret,self.res=cv.threshold(self.res,10,255,cv.THRESH_BINARY_INV)
        #cv.Threshold(self.res, self.res, 10, 255, cv.CV_THRESH_BINARY_INV)
 
    def somethingHasMoved(self):
        nb=0 #Will hold the number of black pixels
        for x in range(self.height): #Iterate the 2hole image
            for y in range(self.width):
                if self.res[x,y] == 0.0: #If the pixel is black keep it
                    nb += 1
        avg = (nb*100.0)/self.nb_pixels #Calculate the average of black pixel in the image

        if avg > self.threshold:#If over the ceiling trigger the alarm
            return True
        else:
            return False
        
if __name__=="__main__":
    detect = MotionDetectorInstantaneous(doRecord=True)
    detect.run()
