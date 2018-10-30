import os
import numpy as np
import cv2

def extract_data(path, images=[]):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.jpeg'):
            image = cv2.imread(abs_path)
            images.append(image)
        
    return images

if __name__ == '__main__':

    folders = ['me', 'other']

    for folder in folders:
        images = extract_data('./cutting_images/{}/'.format(folder), [])
        face_cascade = cv2.CascadeClassifier('C:\\Anaconda\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

        for i, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # saving image
                cv2.imwrite('./cutting_images/cut/{}/image_{}.jpg'.format(folder, i), image[y:y+h, x:x+w])
