import numpy as np
import torch

from .face import Face
from .face_detector import FaceDetector
from .face_recognition import FaceRecognition
from .face_dataset import FaceDataset
from .sort import Sort

class FaceSystem:
    def __init__(self, bio_json_path):

        # Detector & align module
        self.face_detector = FaceDetector(160)

        # Face recognition
        self.face_recognition = FaceRecognition()

        # Dataset faces
        self.dataset = FaceDataset(self, bio_json_path)

        # Tracking system
        self.sort = Sort(max_age=5, min_hits=3)

    def update(self, frame):
        bbs, faces = [], []
        for bb, face in self.face_detector(frame):
            bbs.append(bb)
            faces.append(face)

        if len(bbs) == 0:
            track_results = self.sort.update()
        else:
            track_dets = []
            embs = []
            for bb, emb in zip(bbs, self.face_recognition.get_embbedings(faces)):
                track_dets.append(bb)
                embs.append(emb)

            track_results = self.sort.update(np.array(track_dets), embs)
            
        bbs = []
        mean_embs = []
        for i in range(len(track_results)):
            (x1, y1, x2, y2, _), mean_emb = track_results[i]
            bbs.append((x1, y1, x2, y2))
            mean_embs.append(mean_emb)

        if len(mean_embs) == 0:
            return

        characters = self.face_recognition.get_character_from_embbedings(torch.stack(mean_embs, dim=0), self.dataset)
        for character, (x1, y1, x2, y2) in zip(characters, bbs):
            yield Face((x1, y1, x2, y2), self.dataset[int(character.class_id)] if character is not None else None)
        

    def __call__(self, frame):
        #if True:
        return self.update(frame)
        #else:
        #    return self.predict()

if __name__ == '__main__':
    import time
    import cv2
    fd = FaceSystem()

    im = cv2.imread('../example_2.jpg')

    a0 = time.time()
    for _ in range(15):
        print(list(fd.update(im))[0])

    for _ in range(20):
        print(list(fd.predict()))

    a1 = time.time()
    print(1000 * (a1 - a0))