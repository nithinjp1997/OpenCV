import cv2 as cv
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, modelCom=1, smoothLm=True,
                 enableSeg=False, smoothSeg=True, detectionCon=0.5,
                  trackCon=0.5):
        self.mode = mode
        self.modelCom = modelCom
        self.smoothLm = smoothLm
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, 
                                     self.modelCom, 
                                     self.smoothLm, 
                                     self.enableSeg, 
                                     self.smoothSeg, 
                                     self.detectionCon, 
                                     self.trackCon)
        
    def findPose(self, img, draw=True):
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
            
        return img    
            
    def findPosition(self, img, draw=True):
        lmList = list()
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED) 
        return lmList

def main():
    cap = cv.VideoCapture("PoseDetection\Videos\Vid7.mp4")
    # Get the video's resolution
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Set the window size to match the video's aspect ratio
    aspect_ratio = width / height
    window_height = 600
    window_width = int(window_height * aspect_ratio)
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.resizeWindow('Image', window_width, window_height)
    pTime = 0
    detector = poseDetector()

    while (cap.isOpened()):
        success, img = cap.read()
        if not success:
            # If the video has ended, reset the video to the beginning
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        img = detector.findPose(img) 
        lmList = detector.findPosition(img)
        print(lmList)   
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()