import cv2
import os
import torch
import json
import glob
from PIL import Image

class FaceDataset:
    class Bio:
        COUNTER = 0
        def __init__(self, name, description, thumbnail, photos):
            self.class_id = FaceDataset.Bio.COUNTER
            FaceDataset.Bio.COUNTER += 1
            self.name = name
            self.description = description
            self.thumbnail = thumbnail
            self.photos = photos
            self.embbedings = None

    def __init__(self, face_system, bio_json_path):
        self.photo_folder = os.path.dirname(bio_json_path)
        with open(bio_json_path, 'r') as f:
            data_json = json.load(f)

        self.data = []
        for d in data_json:
            try:
                self.data.append(FaceDataset.Bio(
                    name=d['name'],
                    description=d['description'],
                    thumbnail=cv2.imread(os.path.join(self.photo_folder, d['thumbnail']))[..., :3],
                    photos=[x for photo_path in d['photos'] for x in glob.glob(os.path.join(self.photo_folder, photo_path))]
                ))
            except:
                print(f"Error in {d['thumbnail']}")

        for d in self.data:
            faces = []
            for photo_path in d.photos:
                photo = Image.open(photo_path).convert('RGB')
                aux = list(face_system.face_detector.get_align_face(photo, multiple=False))

                if len(aux) == 0:
                    continue
                _, face = aux[0]
                
                faces.append(face)
            print(f'Computing {d.name} face features...')
            d.embbedings = face_system.face_recognition.get_embbedings(faces)

        self.embbedings = torch.cat([d.embbedings for d in self.data], dim=0)
        self.indices = torch.tensor([i for i, d in enumerate(self.data) for _ in range(d.embbedings.shape[0])])

    def __getitem__(self, idx):
        return self.data[idx]