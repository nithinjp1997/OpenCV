import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS)
        return(img)
    
    def findPosition(self, img, handNo=0, draw=True):

        lmList = list()
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)
        
        return lmList    

def main():
    pTime = 0
    cTime = 0
    
    cap = cv.VideoCapture('HandDetection\Videos\Pexels Videos 2784.mp4')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Set the window size to match the video's aspect ratio
    aspect_ratio = 4/3
    window_height = 500
    window_width = int(window_height * aspect_ratio)
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.resizeWindow('Image', window_width, window_height)
    
    detector = handDetector()

    while True:
        success, img = cap.read()
        
        if not success:
            # If the video has ended, reset the video to the beginning
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0 :
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
        cv.imshow('Image', img)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()        


if __name__ == "__main__":
    main()