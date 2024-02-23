import time

import cv2
import mss
import numpy
from Keyboard import PressKey, ReleaseKey, W, S

#low_chec = numpy.array([80, 100, 100])
#high_chec = numpy.array([100, 255, 255])

low_chec = numpy.array([15, 150, 20])
high_chec = numpy.array([35, 255, 255])

#low_traf = numpy.array([255,0,0])
#high_traf = numpy.array([255,128,0])

with mss.mss() as sct:
    monitor = {
        "left": 419,
        "top": 834,
        "width": 14,
        "height": 21,
    }

    while True:
        last_time = time.time()
        img = numpy.array(sct.grab(monitor))
        cv2.imshow("OpenCV", img)

        #time.sleep(1)
        #PressKey(W)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, low_chec, high_chec)

        moments = cv2.moments(mask, 1)
        dArea = moments['m00']

        i = 0

        if dArea > 5:
            print("чекпоинт")
            d = 11
        '''
            for q in range(15):
                time.sleep(1)
                i += 1
                print(i)
        if i == 15:
            d!=1
            #ReleaseKey(W)
            #PressKey(S)
'''
        #cv2.imshow("HSV", mask)

        if cv2.waitKey(5) == ord("q"):
            cv2.destroyAllWindows()
            break