import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog
)
from PyQt5.QtGui import QMovie, QPixmap
from PyQt5.QtCore import Qt

from connections import ping_connection, download_global, check_blockchain, send_fine_tuned, get_consensus
from predict import model_prediction

URL = "http://127.0.0.1:5000"


class URLInputWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.next_window = None
        self.setWindowTitle("FedExBc")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter server IP")
        self.input_field.setStyleSheet("font-size: 18px; padding: 10px;")
        layout.addWidget(self.input_field)

        # Create a submit button
        submit_button = QPushButton("Connect")
        submit_button.setStyleSheet("font-size: 18px; font-weight: bold;")
        submit_button.clicked.connect(self.handle_submit)  # Connect the button click to the submit function
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def handle_submit(self):
        global URL
        temp = self.input_field.text()
        if temp != "":
            URL = temp
        # ping_connection(URL + "/ping")
        self.next_window = WelcomeWindow()
        self.next_window.show()
        self.close()


class WelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window size
        self.next_window = None
        self.loading = None
        self.setWindowTitle("FedExBc")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()
        layout.setContentsMargins(100, 100, 100, 100)  # Add padding around the window

        heading = QLabel("Fed Ex Bc")
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("font-size: 50px; font-weight: bold;")
        layout.addWidget(heading)

        sub_heading = QLabel("Federated Explainability Blockchain")
        sub_heading.setAlignment(Qt.AlignCenter)
        sub_heading.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(sub_heading)

        spacer = QSpacerItem(0, 200, QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addSpacerItem(spacer)

        paragraph = QLabel()
        paragraph.setText(
            """
            <ul style="font-size: 24px;">
                <li>Our project focuses on connecting various hospitals and their data while maintaining privacy.</li>
                <li>Each hospital connected in the network has access to the global model.</li>
                <li>They can use their private data to update the model without any risk of exposure.</li>
                <li>Trained model is updated on the blockchain with consensus mechanism to validate.</li>
                <li>Our servers federate individual models on having sufficient data and update it on the network.</li>
                <li>We also have a explainability function for the predictions made by the model.</li>
            </ul>
            """
        )
        paragraph.setWordWrap(True)
        paragraph.setAlignment(Qt.AlignLeft)
        paragraph.setStyleSheet("font-size: 24px;")
        layout.addWidget(paragraph)

        spacer = QSpacerItem(0, 200, QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addSpacerItem(spacer)

        button_layout = QHBoxLayout()
        button1 = QPushButton("Download Model")
        button1.clicked.connect(self.download_model)
        button1.setStyleSheet("font-size: 20px; font-weight: bold;")
        button2 = QPushButton("Fine Tune Model")
        button2.clicked.connect(self.model_fine_tune)
        button2.setStyleSheet("font-size: 20px; font-weight: bold;")
        button3 = QPushButton("Make Prediction")
        button3.clicked.connect(self.make_prediction)
        button3.setStyleSheet("font-size: 20px; font-weight: bold;")
        button4 = QPushButton("Validate Blockchain")
        button4.clicked.connect(self.validate_blockchain)
        button4.setStyleSheet("font-size: 20px; font-weight: bold;")
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)
        button_layout.addWidget(button4)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def download_model(self):
        self.loading = LoadingWindow(content="Downloading model ....")
        self.loading.show()
        download_global(URL + "/download")
        self.loading.close()

    def model_fine_tune(self):
        self.next_window = FineTuningWindow()
        self.next_window.show()

    def make_prediction(self):
        dlg = QFileDialog(self, "Select Image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Images (*.jpg *.png *.jpeg)")
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            model_prediction(filenames[0])

    def validate_blockchain(self):
        self.loading = LoadingWindow(content="Validating blockchain ....")
        self.loading.show()
        check_blockchain(URL + "/validate")
        self.loading.close()
        self.loading = LoadingWindow(content="Blockchain validated !!")
        self.loading.show()


class FineTuningWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.loading = None
        self.next_window = None
        self.setWindowTitle("FedExBc")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()
        layout.setContentsMargins(100, 100, 100, 100)  # Add padding around the window

        heading = QLabel("Model Fine Tuning")
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("font-size: 50px; font-weight: bold;")
        layout.addWidget(heading)

        spacer = QSpacerItem(0, 100, QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addSpacerItem(spacer)

        paragraph = QLabel()
        paragraph.setText(
            """
            <ul style="font-size: 24px;">
                <li>Please ensure that the global model has been downloaded first.</li>
                <li>Place the images in a folder named images.</li>
                <li>Store the metadata in a csv file named metadata.csv</li>
                <li>Ensure that the metadata format is as follows:</li>
            </ul>
            """
        )
        paragraph.setWordWrap(True)
        paragraph.setAlignment(Qt.AlignLeft)
        paragraph.setStyleSheet("font-size: 24px;")
        layout.addWidget(paragraph)

        image = QLabel(self)
        pixmap = QPixmap("data_sample.png")
        image.setPixmap(pixmap)
        image.setAlignment(Qt.AlignCenter)
        layout.addWidget(image)

        spacer = QSpacerItem(0, 100, QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addSpacerItem(spacer)

        button_layout = QHBoxLayout()
        button1 = QPushButton("Train Model")
        button1.setStyleSheet("font-size: 20px; font-weight: bold;")
        button2 = QPushButton("Upload Model")
        button2.setStyleSheet("font-size: 20px; font-weight: bold;")
        button2.clicked.connect(self.upload_model)
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def upload_model(self):
        self.loading = LoadingWindow(content="Uploading model ...")
        self.loading.show()
        global URL
        file_name = send_fine_tuned(URL + "/upload")
        reward = get_consensus(URL + "/consensus", file_name)
        self.loading.close()
        self.loading = LoadingWindow(content=f"Received consensus: {reward}")
        self.loading.show()


class LoadingWindow(QWidget):
    def __init__(self, content):
        super().__init__()

        self.setWindowTitle("FedExBc")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        layout.setContentsMargins(100, 100, 100, 100)

        loading_label = QLabel()
        movie = QMovie("loading.gif")
        loading_label.setMovie(movie)
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.resize(50, 50)
        layout.addWidget(loading_label)

        spacer = QSpacerItem(0, 100, QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addSpacerItem(spacer)

        text_label = QLabel(content)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(text_label)

        self.setLayout(layout)
        movie.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = URLInputWindow()
    window.show()
    sys.exit(app.exec_())
