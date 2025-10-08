import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt



# ==== CONFIGURACIÓN ====
ROOT_DIR = r"D:/la-u/ciclo 2025-1/Seminario/DATASET_UNILADO"
OUTPUT_CSV = "./interfaz_etiquetado/etiquetas_por_frame.csv"
ERRORES = [
    "giro_incompleto",
    "pierna_patea_flexionada",
    "pierna_base_flexionada",
    "pie_sin_borde_externo",
    "preparacion_incompleta",
    "recogida_incompleta"
]

# ==== Manejo del CSV ====
if os.path.exists(OUTPUT_CSV):
    etiquetas_df = pd.read_csv(OUTPUT_CSV)
    print(f"✔ Se cargó progreso previo ({len(etiquetas_df)} filas).")
else:
    etiquetas_df = pd.DataFrame(columns=["video", "frame"] + ERRORES)
    etiquetas_df.to_csv(OUTPUT_CSV, index=False)
    print("📄 Se creó nuevo archivo de etiquetas.")

# ==== Funciones auxiliares ====
def obtener_videos_recursivo(root_dir):
    video_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(root, file))
    video_paths.sort()
    return video_paths

def obtener_ultimo_frame(video_name):
    data = etiquetas_df[etiquetas_df["video"] == video_name]
    if len(data) > 0:
        return int(data["frame"].max()) + 1
    return 0

# ==== Clase principal ====
class Etiquetador(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎥 Etiquetador de Errores por Frame")
        self.setGeometry(200, 100, 950, 750)

        self.videos = obtener_videos_recursivo(ROOT_DIR)
        if not self.videos:
            QMessageBox.warning(self, "Error", f"No se encontraron videos en:\n{ROOT_DIR}")
            exit()

        self.current_video_index = 0
        self.current_frame = 0
        self.etiquetas = [0]*len(ERRORES)

        # ==== Layout principal ====
        layout = QVBoxLayout()
        self.label_info = QLabel("Cargando video...")
        self.label_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_info)

        # Mostrar frame
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # ==== Checkboxes ====
        self.checkboxes = []
        check_layout = QVBoxLayout()
        for err in ERRORES:
            cb = QCheckBox(err)
            cb.stateChanged.connect(self.actualizar_etiquetas)
            self.checkboxes.append(cb)
            check_layout.addWidget(cb)
        layout.addLayout(check_layout)

        # ==== Controles ====
        btn_layout = QHBoxLayout()
        self.btn_next = QPushButton("▶️ Siguiente frame (ESPACIO)")
        self.btn_next.clicked.connect(self.next_frame)
        btn_layout.addWidget(self.btn_next)

        self.btn_save = QPushButton("💾 Guardar y salir (S)")
        self.btn_save.clicked.connect(self.save_and_exit)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)

        # ==== Barra de progreso ====
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.setLayout(layout)

        # ==== Cargar primer video ====
        self.load_video()

    def load_video(self):
        if self.current_video_index >= len(self.videos):
            QMessageBox.information(self, "Completado", "✅ Todos los videos han sido etiquetados.")
            self.close()
            return

        self.video_path = self.videos[self.current_video_index]
        self.video_name = os.path.basename(self.video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = obtener_ultimo_frame(self.video_name)

        self.label_info.setText(f"🎬 {self.video_name} - Frame {self.current_frame}/{self.total_frames}")
        self.progress.setMaximum(self.total_frames)
        self.progress.setValue(self.current_frame)
        self.update_frame()

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            self.current_video_index += 1
            self.load_video()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(850, 500, Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        self.label_info.setText(f"{self.video_name} - Frame {self.current_frame}/{self.total_frames}")
        self.progress.setValue(self.current_frame)

    def actualizar_etiquetas(self):
        for i, cb in enumerate(self.checkboxes):
            self.etiquetas[i] = 1 if cb.isChecked() else 0

    def next_frame(self):
        global etiquetas_df
        # Guardar fila actual
        nueva_fila = {"video": self.video_name, "frame": self.current_frame}
        for i, err in enumerate(ERRORES):
            nueva_fila[err] = self.etiquetas[i]
        etiquetas_df = pd.concat([etiquetas_df, pd.DataFrame([nueva_fila])], ignore_index=True)

        # Guardado periódico
        if self.current_frame % 20 == 0:
            etiquetas_df.to_csv(OUTPUT_CSV, index=False)

        # Reset de checkboxes
        for cb in self.checkboxes:
            cb.setChecked(False)

        # Avanzar frame
        self.current_frame += 1
        self.update_frame()

    def save_and_exit(self):
        etiquetas_df.to_csv(OUTPUT_CSV, index=False)
        QMessageBox.information(self, "Guardado", "💾 Progreso guardado correctamente.")
        self.close()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.next_frame()
        elif key == Qt.Key_S:
            self.save_and_exit()
        elif Qt.Key_1 <= key <= Qt.Key_6:
            idx = key - Qt.Key_1
            state = not self.checkboxes[idx].isChecked()
            self.checkboxes[idx].setChecked(state)

# ==== MAIN ====
if __name__ == "__main__":
    app = QApplication([])
    window = Etiquetador()
    window.show()
    app.exec_()
