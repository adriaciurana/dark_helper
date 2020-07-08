#from .align_dlib import AlignDlib
from .facenet_pytorch import MTCNN
import os

# ALIGN_MODEL = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 
#     "weights", 
#     "shape_predictor_68_face_landmarks.dat")

# class FaceDetector:
#     def __init__(self, face_size):
#         self.face_size = face_size
#         self.model = AlignDlib(ALIGN_MODEL, 700)

#     def __call__(self, frame):
#         return self.get_align_face(frame, multiple=True)

#     def get_align_face(self, frame, multiple=True):
#         if multiple:
#             for bb, face in self.model.align_multiple(self.face_size, frame):
#                 yield bb, face
#         else:
#             aux = self.model.align(self.face_size, frame)
            
#             if aux is None:
#                 yield None, None
#                 return

#             bb, face = aux
#             yield bb, face

class FaceDetector:
    def __init__(self, face_size, device='cuda:0'):
        self.face_size = face_size
        self.model = MTCNN(image_size=face_size, margin=0, min_face_size=50, post_process=False, keep_all=True, device=device)


    def __call__(self, frame):
        return self.get_align_face(frame, multiple=True)

    def get_align_face(self, frame, multiple=True):
        faces, bbs = self.model(frame.copy())
        if faces is None:
            return

        if multiple:
            for bb, face in zip(bbs, faces):
                if face is not None:
                    yield bb, face
        else:
            if len(bbs) == 0:
                return

            bb, face = bbs[0], faces[0]
            if face is not None:
                yield bb, face