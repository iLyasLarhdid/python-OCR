import datetime
import os
import threading
import cv2 as cv
# import pytesseract
import easyocr
import numpy as np
import requests

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
cascade = cv.CascadeClassifier("./haar/haarcascade_russian_plate_number.xml")

##### todo : use numpy to store people that we got from the database
##### people = np.array([]) then np.append(people,'user01')
roomName = ""
picturesDir = os.path.join(os.getcwd(), 'pictures')
people = []
studentIds = []
is_entrance = True
platesAlreadyRecognized = []
platesRecognized = {}

# api-endpoint
URL = "http://localhost:8080/ws/cars/speak"
reader = easyocr.Reader(['en', 'ar'])


# thread for sending video
class my_thread(threading.Thread):
    def __init__(self, thread_id, video):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.video = video

    def run(self):
        print("Starting " + str(self.threadID))
        # Get lock to synchronize threads
        threadLock.acquire()
        send_email(self.video)
        # Free lock to release next thread
        threadLock.release()


threadLock = threading.Lock()


def send_email(videoFile):
    print("Video sent.")


def face_rec():
    threads = []
    thread_number = 0
    current_time = ''

    detection = False
    time_detection_started = None
    has_timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 35
    PRECISION = 1500

    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture('testvideo/real3.mp4')
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # movment = cv.VideoWriter("movment.avi", fourcc, 5.0, (1280,720))
    _, frame = cap.read()
    _, frame2 = cap.read()
    licensePlate = ""

    # try:
    #     r = requests.get(url=URL + "/93719l")
    # except:
    #     print("error in request")
    break_detect = False
    while True:
        diff = cv.absdiff(frame, frame2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # print(contours)
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            # print('contors')
            if cv.contourArea(contour) > PRECISION:
                if not detection:
                    detection = True
                    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    # out = cv.VideoWriter(f"{current_time}.mp4", fourcc, 15, frame_size)
                    print("Movement detected... Started Recording!")

        if detection:
            # out.write(frame)
            if has_timer_started:
                tempPlate = extract_plate(frame)
                if len(tempPlate) > 0:
                    licensePlate = tempPlate
                    break_detect = True
                print(time_detection_ends - datetime.datetime.now())
                if datetime.datetime.now() >= time_detection_ends or len(tempPlate) > 0:
                    detection = False
                    has_timer_started = False
                    licensePlate = tempPlate
                    # out.release()
                    # thread_number += 1
                    # print('Stop Recording! sending Plate Number ....!')
                    # # Create new threads
                    # thread = my_thread(thread_number, f"{current_time}.mp4")
                    # # Start new Threads
                    # thread.start()
                    # # Add threads to thread list
                    # threads.append(thread)
                    # # Wait for all threads to complete
                    # print("Exiting Main Thread")
                    # print('Stop Recording!')
                    # sending get request and saving the response as response object
                    try:
                        # print("sending request :" + licensePlate + " --- " + len(licensePlate))
                        if len(licensePlate) > 0:
                            r = requests.get(url=URL + "/" + licensePlate)
                            # extracting data in json format
                            # data = r.json()
                            # print(r)
                    except:
                        print("error in request")
            else:
                # print('now im not 2')
                break_detect = False
                has_timer_started = True
                time_detection_ends = datetime.datetime.now() + datetime.timedelta(
                    seconds=SECONDS_TO_RECORD_AFTER_DETECTION)
                # time_detection_started = time.time()

        frame = frame2
        _, frame2 = cap.read()
        # cv.imshow("Camera", frame)
        # print(facesRecognized)
        # if cv.waitKey(1) == ord('q'):
        #     break

    # out.release()
    cap.release()
    cv.destroyAllWindows()


def extract_plate(img):
    global read
    # img = cv.imread(image_name)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(grey, 1.2, 3)
    licensePlate = ""
    for (x, y, w, h) in nplate:
        # a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        # plate = img[y - 5 + a:(y + 5) + h - a, x - 5 + b:(x + 5) + w - b,:]
        plate = img[y:y + h, x:x + w, :]
        plate = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
        _, plate = cv.threshold(plate, 65, 255, cv.THRESH_BINARY_INV)
        ##cv.imshow("plate", plate)
        # plate = img[y:y + h, x:x + w]
        # kernel = np.ones((1, 1), np.uint8)
        # plate = cv.dilate(plate, kernel, iterations=1)
        # plate = cv.erode(plate, kernel, iterations=1)
        # plate_grey = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
        # (thresh, plate) = cv.threshold(plate_grey, 127, 255, cv.THRESH_BINARY)
        # while True:qqq
        #     cv.imshow("plate", plate)
        #     if cv.waitKey(1) == ord('q'):
        #         break
        ############"""
        # text = pytesseract.image_to_string(plate, lang='eng', config='--psm 6')
        # text = ''.join(e for e in text if e.isalnum())
        text = reader.readtext(plate)
        for txt in text:
            text_bx, plat_num, score = txt
            if score>0.5:
                if plat_num not in platesAlreadyRecognized:
                    platesRecognized.update({plat_num: platesRecognized.setdefault(plat_num, 0) + 1})
                for key, val in list(platesRecognized.items()):
                    if val >= 10:
                        platesAlreadyRecognized.append(key)
                        licensePlate = plat_num
                        #del platesRecognized[key]
                        platesRecognized.clear()
                        print(str(plat_num) + "////" + str(score))
    return licensePlate


face_rec()
