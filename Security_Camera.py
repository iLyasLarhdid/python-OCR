import datetime
import os
import threading
import cv2 as cv
import pytesseract
import numpy as np
import requests

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
cascade = cv.CascadeClassifier("./haar/haarcascade_russian_plate_number.xml")
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv.CascadeClassifier('./haar/cars.xml')

##### todo : use numpy to store people that we got from the database
##### people = np.array([]) then np.append(people,'user01')
roomName = ""
picturesDir = os.path.join(os.getcwd(), 'pictures')
people = []
studentIds = []
is_entrance = True

# api-endpoint
URL = "http://localhost:8080/ws/cars/speak"


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
    SECONDS_TO_RECORD_AFTER_DETECTION = 28
    PRECISION = 1500

    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture('testvideo/real3.mp4')
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # movment = cv.VideoWriter("movment.avi", fourcc, 5.0, (1280,720))
    _, frame = cap.read()
    _, frame2 = cap.read()
    licensePlate = ""
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
            tempPlate = extract_plate(frame)
            if len(tempPlate) > 0:
                licensePlate = tempPlate
            # out.write(frame)
            if has_timer_started:
                print(time_detection_ends - datetime.datetime.now())
                if datetime.datetime.now() >= time_detection_ends:
                    detection = False
                    has_timer_started = False
                    # out.release()
                    thread_number += 1
                    print('Stop Recording! sending email ....!')
                    # Create new threads
                    thread = my_thread(thread_number, f"{current_time}.mp4")
                    # Start new Threads
                    thread.start()
                    # Add threads to thread list
                    threads.append(thread)
                    # Wait for all threads to complete
                    print("Exiting Main Thread")
                    print('Stop Recording!')
                    # sending get request and saving the response as response object
                    try:
                        #print("sending request :" + licensePlate + " --- " + len(licensePlate))
                        if len(licensePlate) > 0:
                            r = requests.get(url=URL + "/" + licensePlate)
                            # extracting data in json format
                            #data = r.json()
                            #print(r)
                    except:
                        print("error in request")
            else:
                # print('now im not 2')
                has_timer_started = True
                time_detection_ends = datetime.datetime.now() + datetime.timedelta(
                    seconds=SECONDS_TO_RECORD_AFTER_DETECTION)
                # time_detection_started = time.time()

        frame = frame2
        _, frame2 = cap.read()
        cv.imshow("Camera", frame)
        # print(facesRecognized)
        if cv.waitKey(1) == ord('q'):
            break

    # out.release()
    cap.release()
    cv.destroyAllWindows()


def extract_plate(img):
    global read
    # img = cv.imread(image_name)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(grey, 1.2, 3)
    licensePlate = ""
    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        car_img = img[y:y+h, x:x + w]
        cv.imshow("cropped", car_img)
        nplate = cascade.detectMultiScale(car_img, 1.1, 4)
        for (x2, y2, w2, h2) in nplate:
            #a2, b2 = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
            #plate = img[y2 - 5 + a2:(y2 + 5) + h2 - a2, x2 - 5 + b2:(x2 + 5) + w2 - b2, :]
            plate = img[y2:y2+h2, x2:x2 + w2]
            cv.imshow("plate", plate)
            kernel = np.ones((1, 1), np.uint8)
            plate = cv.dilate(plate, kernel, iterations=1)
            plate = cv.erode(plate, kernel, iterations=1)
            plate_grey = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
            (thresh, plate) = cv.threshold(plate_grey, 127, 255, cv.THRESH_BINARY)
            cv.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
            # while True:
            #     cv.imshow("plate", plate)
            #     if cv.waitKey(1) == ord('q'):
            #         break
            ############"""
            text = pytesseract.image_to_string(plate, lang='eng', config='--psm 6')
            text = ''.join(e for e in text if e.isalnum())

            print(len(text))
            if 5 <= len(text):
                licensePlate = text
                print("-----------------------" + licensePlate)

    return licensePlate


face_rec()
