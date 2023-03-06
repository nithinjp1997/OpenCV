import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self,mode = False, maxFaces = 2, refineLms = False,
                 minDetectionCon = 0.5, minTrackCon = 0.5, 
                 color = (255, 255, 255), thickness = 2, circle_radius = 2):
        self.mode = mode
        self.maxFaces = maxFaces
        self.refineLms = refineLms
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.refineLms, 
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(self.color, self.thickness, self.circle_radius)

    def findFaceMesh(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = list()
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, 
                                        self.drawSpec, self.drawSpec)
                    face = list()
                    for id,lm in enumerate(faceLms.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                        # cv.putText(img, str(id),(x, y), cv.FONT_HERSHEY_PLAIN,1.3,(0, 255, 0), 2)
                        face.append([x, y])
                faces.append(face)            
        return img, faces                


def main():
    cap = cv.VideoCapture('FaceMesh\Videos\Vid (4).mp4')
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
    detector = FaceMeshDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            # If the video has ended, reset the video to the beginning
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        img, faces = detector.findFaceMesh(img)
        if len(faces) > 0:
            print(faces[0])
        cTime = time.time()
        fps = 1/(cTime- pTime)
        pTime = cTime
        cv.putText(img, f"FPS: {int(fps)}",(20, 70), cv.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()