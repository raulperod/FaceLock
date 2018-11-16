# -*- coding:utf-8 -*-
import cv2
import os
import random
import time
import ctypes

from train import Model
from input import resize_with_pad, write_image
from input import IMAGE_SIZE

DEBUG_OUTPUT = False # Output captured images
CropPadding = 10 # Padding when cropping faces from frames
StrictMode = False
MaxPromptDelay = 1000  # in microsecond
MaxFailDelay = 5000 # in microsecond
SampleInterval = 400 # in microsecond

cascade_path = 'C:\\Anaconda\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'

def extendFaceRect(rect):
    [x, y, w, h] = rect
    if y > CropPadding: y = y - CropPadding
    else: y = 0
    h += 2*CropPadding
    if x > CropPadding: x = x - CropPadding
    else: x = 0
    w += 2*CropPadding
    return [x, y, w, h]

def timestamp():
    return '[' + time.asctime() + ']'

if __name__ == '__main__':
    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    cap = cv2.VideoCapture(0)

    model = Model()
    model.load()

    # Get Cascade Classifier
    cascade = cv2.CascadeClassifier(cascade_path)

    isme=0
    notme=0
    nDelay = 0
    # Run window in other thread
    cv2.startWindowThread()
    for i in range(1000):
        _, frame = cap.read()

        # To gray image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recognize faces
        facerect =  cascade.detectMultiScale(frame_gray, 1.3, 5)

        recStatus = 0
        if len(facerect) > 0:
            print(timestamp(), 'Cara detectada.')
            color = (255, 255, 255)  # 白

            for rect in facerect:
                [x, y, width, height] = extendFaceRect(rect)
                
                # Crop the face
                img_predict = frame[y: y + height, x: x + width]

                # Predict face
                result = model.predict(img_predict)

                if result == 0:  # Is me
                    print(timestamp(), "!Eres tu Raul! :)")
                    isme+=1
                    recStatus = 1
                else:
                    print(timestamp(), 'No eres Raul >:(')
                    notme+=1
                    if recStatus == 0:
                        recStatus = -1

                print(timestamp(), 'yo', isme, 'otro', notme)

        # End if Face Detected
        if recStatus == -1 or (recStatus == 0 and (StrictMode or nDelay >= MaxPromptDelay)):
            nDelay += SampleInterval
            print(timestamp(), 'Ultimo otro hace', nDelay, 's')
        elif recStatus == 1:
            nDelay = 0
            cv2.destroyWindow('Reconociendo.')

        if nDelay >= MaxFailDelay: # Lock Windows
            print(timestamp(), "Bloqueando computadora.")
            ctypes.windll.user32.LockWorkStation()
            nDelay = 0
            cv2.destroyWindow('Reconociendo')
            break
            
        cv2.waitKey(1)
        time.sleep(SampleInterval/500)
           
    # End while True
    # Stop recognize
    cap.release()
    cv2.destroyAllWindows()