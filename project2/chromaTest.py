import chromadb
import json
from deepface import DeepFace
import cv2 as cv

capture = cv.VideoCapture(0)

while(True):
    isFrame , frame = capture.read()
    if not isFrame:
        break
    cv.imshow("Click a picture" , frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        cv.imwrite("image.png",frame)
        cv.destroyAllWindows()
        break
    
print("What is the name : ")
name = input() 
chroma_client = chromadb.PersistentClient(path="./client")
collection = chroma_client.get_or_create_collection(name="face_rec")

embedding = DeepFace.represent(frame, model_name="Facenet")[0]["embedding"]

results = collection.get(include=["embeddings", "documents", "metadatas"])

if len(results['ids']) > 0:
    id = str(int(results['ids'][-1]) + 1)
    print("last id " + id)
else:
    print("No previous entries")
    id = str(1)

collection.upsert(
    ids = [id],
    embeddings = [embedding],
    metadatas=[{"name": name}]
)

