import pandas as pd
import tensorflow as tf
import keras
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtCore import Qt
from main_window import Ui_MainWindow

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from tensorflow.keras.models import load_model


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene_photo = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene_photo)

        self.ui.pushButton.clicked.connect(self.add_photo)

        self.model = load_model('model.keras')


    def add_photo(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_path:
            self.process_image(image_path)

    
    def process_image(self, image_path):
        # Загружаем изображение с помощью OpenCV
        image = cv.imread(image_path)
        if image is None:
            print(f"Ошибка при загрузке изображения: {image_path}")
            return

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("Лица не обнаружены.")
        
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face_resized = cv.resize(face, (160, 190)) 
            face_array = np.expand_dims(face_resized, axis=0)
            face_array = face_array / 255.0

            prediction = self.model.predict(face_array)
            probability = prediction[0][0]
            label = 'Male' if probability > 0.5 else 'Female'

            cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label_text = f"{label} ({probability:.2f})"
            cv.putText(image, label_text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, channels * width, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        self.scene_photo.clear()
        self.scene_photo.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.graphicsView.setScene(self.scene_photo)
        self.ui.graphicsView.fitInView(self.scene_photo.itemsBoundingRect(), Qt.KeepAspectRatio)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


