import matplotlib.pyplot as plt
import mtcnn, cv2, os

classes = os.listdir(f'train/')
for i in classes:
    files = os.listdir(f'train/{i}')
    for j in files:
        face_detector = mtcnn.MTCNN()
        img = plt.imread(f'train/{i}/{j}')
        face = face_detector.detect_faces(img)
        if face:
            x1, y1, width, height = face[0]['box']
            x2, y2 = x1 + width, y1 + height
            img = cv2.resize(img[y1:y2, x1:x2], (224, 224))
            plt.imsave(f'train/{i}/{j}', img)
        else:
            os.remove(f'train/{i}/{j}')
classes = os.listdir(f'valid/')
for i in classes:
    files = os.listdir(f'valid/{i}')
    for j in files:
        face_detector = mtcnn.MTCNN()
        img = plt.imread(f'valid/{i}/{j}')
        face = face_detector.detect_faces(img)
        if face:
            x1, y1, width, height = face[0]['box']
            x2, y2 = x1 + width, y1 + height
            img = cv2.resize(img[y1:y2, x1:x2], (224, 224))
            plt.imsave(f'valid/{i}/{j}', img)
        else:
            os.remove(f'valid/{i}/{j}')
