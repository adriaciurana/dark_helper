from capture_monitor import CaptureMonitor
from face_lib import FaceSystem
from visualizer import Visualizer #MainWindow
#from PyQt5 import QtCore, QtGui, QtWidgets
import os

monitor = CaptureMonitor(bb=(0, 0, 600, 480))
face_system = FaceSystem(os.path.abspath("./bio/bio.json"))

def thread_process():
	frame = monitor.get_frame()

	faces = []
	for face in face_system(frame):
		faces.append(face)

	return frame, faces

if __name__ == '__main__':
	# app = QtWidgets.QApplication([])
	# window = MainWindow()
	# window.process(thread_process)

	# window.show()
	# app.exec_()
	app = Visualizer(monitor, thread_process)
	app.run()