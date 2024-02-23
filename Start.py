import pytesseract
from PIL import ImageGrab
import time
import numpy as np
import cv2
import mss
from Keyboard import PressKey, ReleaseKey, W, S
import pyautogui as pg
import tensorflow as tf
from keras.models import load_model
import os


#model = tf.keras.models.load_model('modelsmta.hdf5')   #python 3.10.0
model = load_model('modelsmta.hdf5', compile = False)   #python 3.7

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

filename = 'image.png'
filename11 = 'da.png'

low_chec = np.array([15, 150, 20])
high_chec = np.array([35, 255, 255])

index = 1
index1 = 1
index11 = 1
index2 = 100

with mss.mss() as sct:
    monitor = {
        "left": 419,
        "top": 815,
        "width": 14,
        "height": 21,
    }

    while True:

        screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(1517, 839, 1554, 908))), cv2.COLOR_BGR2RGB)  # Скорость
        #cv2.imshow('window12', screenshot)

        img = cv2.resize(screenshot, (100, 100))
        img = np.array(img)
        img = img / 255.0

        prediction = model.predict(img.reshape(1, 100, 100, 3))
        digit = np.argmax(prediction)
        #print('цифра: ', digit)

        screen = np.array(ImageGrab.grab(bbox=(860, 820, 1060, 910)))  # Двери2
        screen11 = np.array(ImageGrab.grab(bbox=(933, 629, 960, 647)))  # Диалоговое окно
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        img1 = np.array(sct.grab(monitor))  #sct
        #cv2.imshow('window', img1)
        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low_chec, high_chec)
        moments = cv2.moments(mask, 1)
        dArea = moments['m00']
        d = dArea > 5
        d == 1

        if digit <= 1:
            cv2.imwrite(filename, gray) # Записывает двери открыты/закрыты
            img = cv2.imread('image.png')   # Читает двери открыты/закрыты

            cv2.imwrite(filename11, screen11)   #Записывает да
            img11 = cv2.imread('da.png')    # Читает да
            #text = pytesseract.image_to_string(img11, lang='rus')
            text = pytesseract.image_to_string(img11)
            index11 = text.find("Да")
            #print(index11)
            text = pytesseract.image_to_string(img, lang="rus")
            #print(text)
            index = text.find("подождите")
            #print(index)
            index1 = text.find("Закрой")
            #print(index1)
            index2 = text.find("Заработано")
            #print(index2)
        if digit == 1:
            monitor = {
                "left": 419,
                "top": 834, #834
                "width": 14,
                "height": 21,
            }
            #print('Скорость 10 км/ч')
        elif digit ==2:
            monitor = {
                "left": 419,
                "top": 832, #832
                "width": 14,
                "height": 21,
            }
            #print('Скорость 20 км/ч')
        elif digit == 3:
            monitor = {
                "left": 419,
                "top": 828, #828
                "width": 14,
                "height": 21,
            }
            #print('Скорость 30 км/ч')
        elif digit ==4:
            monitor = {
                "left": 419,
                "top": 815, #815
                "width": 14,
                "height": 21,
            }
            #print('Скорость 40 км/ч')
        elif digit == 5:
            monitor = {
                "left": 419,
                "top": 810,  # 810
                "width": 14,
                "height": 21,
            }
            #print('Скорость 50 км/ч')

        if d!=1:  # если не видишь чекпоинт, тогда едешь вперед
            ReleaseKey(S)  # попробовать сначало это
            PressKey(W)
            #print("Не вижу чекпоинт")
        elif d==1 and index1 == 0:  #если видишь чекпоинт и видишь "закройте двери"
            PressKey(W)
            PressKey(S)

            time.sleep(1)
            PressKey(3)
            time.sleep(1)
            ReleaseKey(3)

            print("Закрываю двери")
            time.sleep(4)
        elif d==1 and index == 18:  #если видишь чекпоинт и "откройте двери"
            PressKey(W)
            PressKey(S)

            time.sleep(1)
            PressKey(3)
            time.sleep(1)
            ReleaseKey(3)

            print('Открываю двери')
            time.sleep(5)
        elif d==1:  #если видишь чекпоинт, тогда тормозишь
            PressKey(S)
            ReleaseKey(W)  #потом попробовать это
            #print("Вижу чекпоинт, торможу")

        if index11 == 0:  # диалоговое окно
            pg.click(959,638)
            time.sleep(5)
            PressKey(W)
            PressKey(S)
            ReleaseKey(W)
            ReleaseKey(S)
            print('Продолжаю поездку')

        if index2 == 1:
            print(text)

#

        if cv2.waitKey(5) == ord("q"):
            cv2.destroyAllWindows()
            break