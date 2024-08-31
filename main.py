import cv2 as cv
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from main_window import Ui_MainWindow
from tensorflow.keras.preprocessing import image
import tensorflow as tf


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.label.setWordWrap(True)

        self.scene_photo = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene_photo)

        self.ui.pushButton.clicked.connect(self.add_photo)

        self.model = tf.keras.models.load_model('model.keras')

    def add_photo(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_path:
            self.process_image(image_path)
            self.display_original_image(image_path)

    def process_image(self, image_path):
        try:
            img = image.load_img(image_path)

        except Exception as e:
            print(f"Error to download: {image_path}\nОшибка: {e}")
            return
        
        target_size = (160, 190)
        img = img.resize(target_size)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = self.model.predict(img_array)

        probability_class_1 = prediction[0][0]
        probability_class_0 = 1 - probability_class_1

        self.ui.label.setText(f'Probability (man): {probability_class_1 * 100:.2f}%\n'
                              f'Probability (woman): {probability_class_0 * 100:.2f}%\n')

    def display_original_image(self, image_path):
        try:
            img = cv.imread(image_path)
            if img is None:
                print(f"Error to download: {image_path}")
                return
        except Exception as e:
            print(f"Error to download: {e}")
            return

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        height, width, channels = img_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)

        self.scene_photo.clear()
        self.scene_photo.addItem(QGraphicsPixmapItem(pixmap))

        self.ui.graphicsView.setScene(self.scene_photo)
        self.ui.graphicsView.fitInView(self.scene_photo.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.display_image_with_face_box(image_path)


    def display_image_with_face_box(self, image_path):
        try:
            img = cv.imread(image_path)
            if img is None:
                print(f"Error: {image_path}")
                return
        except Exception as e:
            print(f"Error: {e}")
            return

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error to download cascade.")
            return

        faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1) 

        height, width, channels = img_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

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


