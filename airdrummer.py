## May 25 2022 Robert Hochgraf


# First import the libraries
import numpy as np
# import pandas as pd
import cv2
import mediapipe as mp

# print copyright/debug/system information
import time
import mido

# A PoseDetector is an object that represents a pose of a body, stored by the positions of its key points
class PoseDetector:

    # constructs a PoseDetector
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=False, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # self.mpHolistic= mp.solutions.holistic

        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        for landmark in self.mpPose.PoseLandmark:
            print(landmark)

    # finds a pose from an image and draws key points on it
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    # returns the positions of the landmarks; draws circles at the points
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    # function returns a vector representing the left hand position
    def getLeftHand(self, img):
        if self.results.pose_landmarks:
            lWrist = [self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST.value].x,
                      self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST.value].y,
                      self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST.value].z]
            lWrist = np.array(lWrist)
            return lWrist

    # returns a vector3 representing the right hand position
    def getRightHand(self, img):
        if self.results.pose_landmarks:
            rWrist = [self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST.value].x,
                      self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST.value].y,
                      self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST.value].z]
            rWrist = np.array(rWrist)
            return rWrist

    # returns a vector3 representing the average shoulder position
    def getAvgShoulder(self, img):
        if self.results.pose_landmarks:
            shoulderL = [self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                         self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].y,
                         self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].z]
            shoulderL = np.array(shoulderL)
            shoulderR = [self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].y,
                         self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].z]
            shoulderR = np.array(shoulderR)
            shoulderM = (shoulderL + shoulderR) * 0.5
            return shoulderM

''' Overlays the overlay image on the background, x, y are offset for where top left of image is placed.
    Cuts off the overlay image if it does out of range of the background.'''
def overlay_transparent(background, overlay, x, y):
    x = int(x)
    y = int(y)

    # remove the alpha channel from the overlay
    overlay_3_ch = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)

    # store overlay shape
    h, w, c = overlay.shape

    # store background shape
    h2, w2, c2 = background.shape

    # get legal placement area
    x_max = min(x+w, x+w2)
    y_max = min(y+h, y+h2)

    if x < 0:
        xdiff = -x
    else:
        xdiff = 0
    if y < 0:
        ydiff = -y
    else:
        ydiff = 0

    x = max(x,0)
    y = max(y,0)

    # get the background within the area we will place the overlay
    background_scaled = background[x:x_max, y:y_max]

    # get the scaled shape
    xmax = background_scaled.shape[0]
    ymax = background_scaled.shape[1]

    # make sure the overlay fits within background
    overlay_3_ch = overlay_3_ch[xdiff:xmax+xdiff, ydiff:ymax+ydiff]

    # get the alpha channel from the overlay
    a = overlay[xdiff:xmax+xdiff, ydiff:ymax+ydiff, 3]
    a = cv2.merge([a, a, a])

    # blend the two images using the alpha channel as controlling mask
    result = np.where(a == (0, 0, 0), background_scaled, overlay_3_ch)

    background[x:x_max, y:y_max] = result[:, :]

    return background


# The main loop captures video from the webcam and draws poses
def main():
    # variables for quick customization
    sleep_time = 0  # how long to sleep before running
    record_time = 45   # how long for the camera to record data before quitting

    # initialize the camera and the program
    time.sleep(sleep_time)
    cap = cv2.VideoCapture(0)  # make VideoCapture(0) for webcam
    start_time = time.time()

    # initialize midi output
    outport = mido.open_output(name='webcamMidi', virtual=True)

    # initialize a PoseDetector
    detector = PoseDetector()

    # store if previous value was a trigger
    triggerL = False
    triggerR = False

    prevTime = 0

    # while less than the runtime, read images from the camera and print output about it
    while (time.time() < (start_time + record_time)):  # collect for 20 seconds
        # read the image from the camera
        success, img = cap.read()
        # draw the pose on the image
        img = detector.findPose(img)
        # update time info
        currTime = time.time()
        fps = 1 / (currTime-prevTime)
        prevTime = currTime

        # trigger an event if the left hand is higher than the avg shoulder position
        lHand = detector.getLeftHand(img)
        rHand = detector.getRightHand(img)
        mShoulder = detector.getAvgShoulder(img)

        # left hand plays bass drum and right plays snare
        lNote = 36
        rNote = 38

        if(lHand[1] > mShoulder[1] and triggerL == False):
            triggerL = True
            msg = mido.Message('note_on', note=lNote)
            outport.send(msg)
        if(lHand[1] <= mShoulder[1]):
            triggerL = False
            msg = mido.Message('note_off', note=lNote)
            outport.send(msg)

        if(rHand[1] > mShoulder[1] and triggerR == False):
            triggerR = True
            msg = mido.Message('note_on', note=rNote)
            outport.send(msg)
        if(rHand[1] <= mShoulder[1]):
            triggerR = False
            msg = mido.Message('note_off', note=rNote)
            outport.send(msg)

        # display info about the left hand
        cv2.putText(img, ('LH X Pos:' + str(lHand[0])), (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, ('LH Y Pos:' + str(lHand[1])), (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, ('LH Z Pos:' + str(lHand[2])), (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # draw a drum set on the image
        # IMREAD_UNCHANGED keeps the transparency of the png
        overlay = cv2.imread('drumset.png', cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, (500,500))

        # resize the images
        img_scaled = cv2.resize(img, (1280, 960))

        # merge the two images
        ypos = mShoulder[0] * img_scaled.shape[0]
        xpos = mShoulder[1] * img_scaled.shape[1]
        img_scaled = overlay_transparent(img_scaled, overlay, xpos, ypos)

        # display the resized image
        cv2.imshow("Image", img_scaled)

        # sleep cv2 for 1 ms
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
