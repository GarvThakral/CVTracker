from deepface import DeepFace
import cv2
import chromadb 
from numpy import dot
from numpy.linalg import norm


chroma_client = chromadb.PersistentClient(path="./client")

collection = chroma_client.get_collection("face_rec")

detName = ""

cap = cv2.VideoCapture(0)
frameIndex = 0
while True:
    frameIndex += 1
    ret, frame = cap.read()
    

    if not ret:
        break
    if(frameIndex % 100 == 0):
        faces = DeepFace.extract_faces(frame , enforce_detection=False)
        if isinstance(faces, list) and len(faces) > 0:
            fa = faces[0]["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
            print(faces)
            face_crop = frame[y:y+h, x:x+w]
            embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            results = collection.get(include = ["embeddings","metadatas"])
            idArr = results['ids']
            embeddingArr = results['embeddings']
            metadatasArr = results['metadatas']
            print("Names")
            print(metadatasArr)
            maxSim = 0
            detId = None
            faces = None
                    
            for i , e in enumerate(embeddingArr):
                cos_sim = dot(e, embedding)/(norm(e)*norm(embedding))
                if(cos_sim > maxSim):
                    maxSim = cos_sim
                    detName = metadatasArr[i]['name']
                    detId = idArr[i]
            if(maxSim > 0.6):
                pass
            else:
                detName = ""
        if frameIndex == 400:
            frameIndex = 0
    cv2.putText(frame , "Detected : " + detName , (0,250),
        cv2.FONT_HERSHEY_TRIPLEX , 1.0 , (0,255,0) , 2)

    cv2.imshow("frame", frame)


   
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
cap.release()
cv2.destroyAllWindows()
