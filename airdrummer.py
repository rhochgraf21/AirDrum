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
            print("DRUM TRIGGER")
            msg = mido.Message('note_on', note=lNote)
            outport.send(msg)
        if(lHand[1] <= mShoulder[1]):
            triggerL = False
            msg = mido.Message('note_off', note=lNote)
            outport.send(msg)

        if(rHand[1] > mShoulder[1] and triggerR == False):
            triggerR = True
            print("DRUM TRIGGER")
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

        # resize the image
        img_scaled = cv2.resize(img, (1280, 960))

        # display the resized image
        cv2.imshow("Image", img_scaled)
        # sleep cv2 for 1 ms
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
