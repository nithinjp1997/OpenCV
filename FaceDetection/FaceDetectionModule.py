import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5, modelSel=0):
        self.minDetectionCon = minDetectionCon
        self.modelSel = modelSel
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon,
                                                                self.modelSel)
    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = list()

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv.putText(img,f"{int(detection.score[0]*100)}%", 
                            (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 255), 3)

        return img, bboxs
    
    def fancyDraw(self, img, bbox, l=90, t=10, rt=2):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv.rectangle(img, bbox, (255, 0 ,255), rt)
        # Top Left x,y
        cv.line(img,(x, y), (x+l, y), (255, 0, 255), t)
        cv.line(img,(x, y), (x, y+l), (255, 0, 255), t)
        # Top Right x1,y
        cv.line(img,(x1, y), (x1-l, y), (255, 0, 255), t)
        cv.line(img,(x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left x,y1
        cv.line(img,(x, y1), (x+l, y1), (255, 0, 255), t)
        cv.line(img,(x, y1), (x, y1-l), (255, 0, 255), t)
        # Bottom Right x1,y1
        cv.line(img,(x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv.line(img,(x1, y1), (x1, y1-l), (255, 0, 255), t)

        return img

    
def main():
    cap = cv.VideoCapture('FaceDetection\Videos\Vid (2).mp4')
    # Get the video's resolution
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Set the window size to match the video's aspect ratio
    aspect_ratio = width/height
    window_height = 600
    window_width = int(window_height * aspect_ratio)
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.resizeWindow('Image', window_width, window_height)

    pTime = 0
    detector = FaceDetector()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            # If the video has ended, reset the video to the beginning
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        
        img, bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img,f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 0), 3)
        cv.imshow("Image",img)    
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()