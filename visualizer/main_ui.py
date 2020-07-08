import eel

import json

import cv2
import base64
import numpy as np
from PIL import Image as PILlib
import io
import re
import sys, os

from threading import Thread

class Images:
    @staticmethod
    def read(im_path):
        return cv2.imread(im_path)

    @staticmethod
    def unpack_im(pack, image_type):
        if image_type == 'numpy':
            b64, dtype, shape = pack
            return np.frombuffer(base64.decodebytes(b64.encode()), dtype=dtype).reshape(shape)
        
        elif image_type == 'jpeg' or image_type == 'jpg':
            m = re.search(r'base64,(.*)', pack)
            if m is None:
                raise IndexError

            imgstring = m.group(1)
            
            # aplicamos una correccion para evitar un error de padding
            imgbyte = imgstring.encode()
            pad = len(pack.partition(",")[2]) % 4
            imgbyte += b"="*pad
            image = cv2.imdecode(np.frombuffer(io.BytesIO(base64.b64decode(imgbyte)).getbuffer(), np.uint8), -1)
            
            return image[..., :3]

    @staticmethod
    def pack_im(im, image_type):
        if image_type == 'numpy':
            return base64.b64encode(np.ascontiguousarray(im)).decode(), im.dtype.name, im.shape

        elif image_type == 'jpeg' or image_type == 'jpg':
            return 'data:image/jpg; base64,' + base64.b64encode(cv2.imencode('.jpg', im)[1]).decode("utf-8")

class Visualizer():
    def __init__(self, monitor, callback):
        self.monitor = monitor
        self.callback = callback
        self.is_close = False

    def close(self, *vargs):
        self.is_close = True

    def run(self):
        eel.init(os.path.join(os.path.dirname(os.path.abspath(__file__))))

        @eel.expose
        def set_settings(data):
            data = json.loads(data)

            self.monitor.set_mode(int(data['mode']))
            self.monitor.set_window(data['window'])
            self.monitor.change_coords(int(data['left']), 
                int(data['top']), 
                int(data['width']), 
                int(data['height']))

        @eel.expose
        def get_settings():
            params = {
                'left': self.monitor.current_monitor['left'],
                'top': self.monitor.current_monitor['top'],
                'width': self.monitor.current_monitor['width'],
                'height': self.monitor.current_monitor['height'],

                'mode': self.monitor.mode,
                'windows': self.monitor.get_windows(),
                'current_window': self.monitor.current_window.get_wm_name() if self.monitor.current_window is not None else None
            }

            print(params)

            return json.dumps(params)

        def __run():
            while not self.is_close:
                frame, faces = self.callback()

                frame = frame[..., ::-1].copy()
                for face in faces:
                    x1, y1, x2, y2 = face.bb
                    if face.character is not None:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, -1)
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, -1)

                unique_bios = set()
                for face in faces:
                    if face.character is not None:
                        unique_bios.add(face.character)

                data = {
                    'image_detection': Images.pack_im(frame, 'jpg'),
                    'bios': [{'photo': Images.pack_im(character.thumbnail, 'jpg'), 'name': character.name, 'description': character.description} for character in unique_bios]
                }
                data_json = json.dumps(data)
                eel.draw(data_json)
        Thread(target=__run).start()
        eel.start('main.html', mode='chrome', port=0, size=(520, 1280), close_callback=self.close)


