### 🔍 **What the project does**

This project is designed to **automatically detect vehicle license plates** from a video feed (or webcam), **read the text on the plate using OCR**, and **send the plate number to an API** once the plate is confidently recognized.

---

### 🚦 **How it works in simple steps**

1. **Starts video capture**:  
   The program uses a pre-recorded video (or webcam) as input.

2. **Detects movement**:  
   It continuously compares frames to check if there’s motion (e.g. a car entering the scene).

3. **Starts scanning when motion is detected**:  
   Once movement is spotted, the system begins monitoring closely for a short period (about 35 seconds) to detect a license plate.

4. **Detects license plates visually**:  
   It uses a trained AI model (Haar Cascade) to find the area in the frame where a license plate might be.

5. **Reads the text on the plate**:  
   The detected plate area is passed through an OCR engine (EasyOCR), which reads and extracts the characters.

6. **Verifies plate recognition**:  
   To avoid false positives, the program waits until the same plate is recognized several times with good confidence before considering it valid.

7. **Sends plate to server**:  
   Once confirmed, it sends the license plate number to a predefined API endpoint for further processing (like logging entry, checking database, etc.).

8. **Optional threading for video recording (disabled)**:  
   There’s a placeholder for saving video and sending it in a background thread, but it's currently inactive.

---

### ✅ **Main goal**

To automate **vehicle license plate recognition from video**, and send this information to a backend system reliably and with confidence.
