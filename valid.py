import cv2
import numpy as np
import random

from train import Model

from input import extract_data, write_image
from input import DEBUG_OUTPUT, GRAY_MODE
# from image_show import show_image

DEBUG_MUTE = False # Stop outputing unnecessary infomation
DEBUG_OUTPUT = False # Output processed images
CropPadding = 10 # Padding when cropping faces from frames

test_path = 'C:/Proyectos/FaceLock/data/valid'

def extendFaceRect(rect):
    [x, y, w, h] = rect
    if y > CropPadding: y = y - CropPadding
    else: y = 0
    h += 2*CropPadding
    if x > CropPadding: x = x - CropPadding
    else: x = 0
    w += 2*CropPadding
    return [x, y, w, h]

if __name__ == '__main__':
    model = Model()
    model.load()

    # Get Cascade Classifier
    cascade = cv2.CascadeClassifier('C:\\Anaconda\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

    images, labels = extract_data(test_path)    
    labels = np.reshape(labels, [-1])

    right = 0
    cross = 0
    countme = 0
    countnotme = 0
    rightme = 0
    rightnotme = 0
    
    for idx, image in enumerate(images):
        # To gray image
        if GRAY_MODE is False:
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = image

        # Recognize faces
        #facerect = cascade.detectMultiScale(frame_gray, 1.3, 5)
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))

        if len(facerect) > 0:
            if DEBUG_MUTE == False:
                print('face detected')
            color = (255, 255, 255)  # 白
            for (x, y, w, h) in facerect:
                [x, y, w, h] = extendFaceRect([x, y, w, h])
                img_predict = image[y: y + h, x: x + w]

                if GRAY_MODE == True:
                    result = model.predict(img_predict, img_channels=1)
                else:
                    result = model.predict(img_predict)
                if DEBUG_MUTE == False:
                    print(labels[idx])
                if result == labels[idx]:  # right
                    right += 1
                else:
                    cross += 1

                if labels[idx] == 0: #isme
                    countme += 1
                    if result == 0:
                        rightme += 1
                    elif DEBUG_OUTPUT == True :
                        write_image('./output/mistake/' + str(random.randint(1,999999)) + '.jpg', image)
                else:
                    countnotme += 1
                    if result == 1:
                        rightnotme += 1
                    elif DEBUG_OUTPUT == True :
                        write_image('./output/mistake/' + str(random.randint(1,999999)) + '.jpg', image)

    print('right: ', right, 'false: ', cross)
    print('countme: ', countme, 'rightme: ', rightme)
    print('countnotme: ', countnotme, 'rightnotme: ', rightnotme)
    print('accuracy: ', (float)(right/(right+cross)))

    # Stop recognize
    cv2.destroyAllWindows()
    