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
print("What is the rollNum : ")
rollNum = input()
chroma_client = chromadb.PersistentClient(path="./client")
collection = chroma_client.get_or_create_collection(name="face_rec")

embedding = DeepFace.represent(frame, model_name="Facenet")[0]["embedding"]

results = collection.get(include=["embeddings", "documents", "metadatas"])
if len(results['ids']) > 0:
    print(results['ids'][-1])
else:
    print("No previous entries")
    id = 1
    
collection.upsert(
    ids = [rollNum],
    embeddings = [embedding],
    metadatas=[{"name": name}]
)


with open("test.json",'a') as f:
    f.write(str(results))