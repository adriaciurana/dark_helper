import torch
from .facenet_pytorch import InceptionResnetV1

class FaceRecognition:
    def __init__(self, device='cuda:0'):
        self.model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
        self.device = device
        self.min_dist = 1.1

    def fixed_image_standardization(self, image_tensor):
        processed_tensor = (image_tensor - 127.5) / 128.0
        return processed_tensor

    def get_embbedings(self, faces):
        with torch.no_grad(): #torch.from_numpy(face).permute(2, 0, 1).to(self.device).float()
            faces = torch.stack([face for face in faces], dim=0).to(self.device)
            faces = self.fixed_image_standardization(faces)

            return self.model(faces)

    def get_character_from_embbedings(self, embs, dataset):
        dataset_embbedings = dataset.embbedings
        
        distances = torch.cdist(embs.unsqueeze(0), dataset_embbedings.unsqueeze(0), p=2)[0]
        real_indices = dataset.indices

        characters_idxs = torch.unique(real_indices)

        character_distance_matrix = (self.min_dist + 1) * torch.ones((embs.shape[0], characters_idxs.shape[0]))
        for i in characters_idxs:
            mask = real_indices == i
            character_distance_matrix[:, i] = torch.mean(distances[:, mask], dim=1)

        min_values, indices = torch.min(character_distance_matrix, axis=1)
        return [dataset[idx]  if min_value < self.min_dist else None for min_value, idx in zip(min_values, indices)]
