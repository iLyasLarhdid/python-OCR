import cv2 as cv
import pytesseract
import numpy as np
########---------------------tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'
####-------------------------text = pytesseract.image_to_string(img, config=tessdata_dir_config)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
cascade = cv.CascadeClassifier("./haar/haarcascade_russian_plate_number.xml")


def extract_plate(image_name):
    print("hello world")
    global read
    img = cv.imread(image_name)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(grey, 1.1, 4)
    for (x, y, w, h) in nplate:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        plate = img[y-5 + a:(y+5) + h - a, x-5 + b:(x+5) + w - b, :]
        kernel = np.ones((1, 1), np.uint8)
        plate = cv.dilate(plate, kernel, iterations=1)
        plate = cv.erode(plate, kernel, iterations=1)
        plate_grey = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
        (thresh, plate) = cv.threshold(plate_grey, 127, 255, cv.THRESH_BINARY)
        while True:
            cv.imshow("plate", plate)
            if cv.waitKey(1) == ord('q'):
                break
        ############"""
        text = pytesseract.image_to_string(plate, lang='eng', config='--psm 6')
        text = ''.join(e for e in text if e.isalnum())
        print(text)


extract_plate("./testimages/t6.jpg")
