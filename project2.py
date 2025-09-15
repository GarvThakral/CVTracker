import cv2 as cv
from insightface.app import FaceAnalysis

# Init camera
capture = cv.VideoCapture(0)

# Init InsightFace
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    faces = app.get(frame)  

    for face in faces:

        (x, y, x2, y2) = face.bbox.astype(int)
        cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)


        embedding = face.embedding 
        print("Embedding:", embedding[:5], "...")  

    cv.imshow("Final", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
