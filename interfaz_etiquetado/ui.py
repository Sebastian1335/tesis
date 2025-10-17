import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QMessageBox, QProgressBar, QListWidget
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

# ==== FUNCIONES AUXILIARES ====
def obtener_videos_recursivo(root_dir):
    video_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(root, file))
    video_paths.sort()
    return video_paths

# lista global de videos (orden fijo)
VIDEO_PATHS = obtener_videos_recursivo(ROOT_DIR)

def guardar_csv_ordenado():
    """
    Guarda etiquetas_df ordenado por el orden fijo VIDEO_PATHS y frame.
    Evita duplicados por (video,frame).
    """
    global etiquetas_df
    if etiquetas_df.empty:
        etiquetas_df.to_csv(OUTPUT_CSV, index=False)
        return

    categorias = [os.path.basename(v) for v in VIDEO_PATHS]
    etiquetas_df["video"] = pd.Categorical(etiquetas_df["video"], categories=categorias, ordered=True)

    etiquetas_df = (
        etiquetas_df.sort_values(["video", "frame"])
        .drop_duplicates(subset=["video", "frame"], keep="last")
        .reset_index(drop=True)
    )

    etiquetas_df.to_csv(OUTPUT_CSV, index=False)

# ==== Manejo del CSV ====
if os.path.exists(OUTPUT_CSV):
    etiquetas_df = pd.read_csv(OUTPUT_CSV)
    print(f"✔ Se cargó progreso previo ({len(etiquetas_df)} filas).")
else:
    etiquetas_df = pd.DataFrame(columns=["video", "frame"] + ERRORES)
    guardar_csv_ordenado()
    print("📄 Se creó nuevo archivo de etiquetas.")


def obtener_ultimo_frame(video_name):
    """
    Devuelve el primer frame NO etiquetado para ese video.
    Si todos están etiquetados devuelve total_frames (manejamos eso en load_video).
    """
    data = etiquetas_df[etiquetas_df["video"] == video_name]
    if len(data) == 0:
        return 0
    # buscar el siguiente índice disponible: max(frame) + 1
    return int(data["frame"].max()) + 1


# ==== VENTANA DE ETIQUETADO ====
class Etiquetador(QWidget):
    def __init__(self, video_path, revision_mode=False):
        super().__init__()
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.revision_mode = revision_mode
        self.setWindowTitle(f"🎥 Etiquetador - {self.video_name}")
        self.setGeometry(200, 100, 1000, 800)
        self.etiquetas = [0] * len(ERRORES)
        self.load_data()

        # Layout
        layout = QVBoxLayout()
        self.label_info = QLabel("Cargando video...")
        self.label_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_info)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Checkboxes
        self.checkboxes = []
        check_layout = QHBoxLayout()
        for err in ERRORES:
            cb = QCheckBox(err)
            cb.stateChanged.connect(self.actualizar_etiquetas)
            self.checkboxes.append(cb)
            check_layout.addWidget(cb)
        layout.addLayout(check_layout)

        # Botones
        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⏮ Frame anterior (←)")
        self.btn_prev.clicked.connect(self.prev_frame)
        btn_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("⏭ Siguiente frame (→ / ESPACIO)")
        self.btn_next.clicked.connect(self.next_frame)
        btn_layout.addWidget(self.btn_next)

        self.btn_save = QPushButton("💾 Guardar y salir (S)")
        self.btn_save.clicked.connect(self.save_and_exit)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)

        # Progreso
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.setLayout(layout)

        # Abrir video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"No se pudo abrir el video:\n{self.video_path}")
            self.close()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Si venimos del modo normal calculamos start_frame
        start_frame = obtener_ultimo_frame(self.video_name)
        if not self.revision_mode and start_frame >= self.total_frames:
            frames_existentes = etiquetas_df.loc[
                (etiquetas_df["video"] == self.video_name), "frame"
            ].tolist()

            if len(frames_existentes) == 0:
                # ningún frame etiquetado (nuevo)
                self.current_frame = max(self.total_frames - 1, 0)
            else:
                # permitir revisión aunque todas las etiquetas sean 0
                self.revision_mode = True
                self.frames_revisar = sorted(frames_existentes)
                self.current_frame = 0
                QMessageBox.information(
                    self, "Info",
                    f"Video completamente etiquetado ({len(frames_existentes)} frames). Abriendo en modo revisión."
                )
        else:
            # modo normal
            self.current_frame = min(start_frame, max(self.total_frames - 1, 0))
            self.frames_revisar = sorted(
                etiquetas_df.loc[
                    (etiquetas_df["video"] == self.video_name) & (etiquetas_df[ERRORES].sum(axis=1) > 0), "frame"
                ].tolist()
            ) if self.revision_mode else []

        self.video_data = etiquetas_df[etiquetas_df["video"] == self.video_name]
        self.update_frame()

    def load_data(self):
        # placeholder (si más adelante quieres cargar metadata)
        pass

    def update_frame(self):
        # elegir número de frame a mostrar
        if self.revision_mode:
            if not self.frames_revisar:
                QMessageBox.information(self, "Sin etiquetas", "No hay frames etiquetados en este video.")
                self.close()
                return
            # clamp current_frame índice en frames_revisar
            self.current_frame = max(0, min(self.current_frame, len(self.frames_revisar) - 1))
            frame_number = self.frames_revisar[self.current_frame]
        else:
            # clamp con total_frames
            self.current_frame = max(0, min(self.current_frame, max(self.total_frames - 1, 0)))
            frame_number = self.current_frame

        # intentar leer
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            # intento de recuperación: si frame_number >= total_frames intentar con total_frames-1
            if frame_number >= self.total_frames and self.total_frames > 0:
                recovery = max(self.total_frames - 1, 0)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, recovery)
                ret, frame = self.cap.read()
                if not ret:
                    QMessageBox.warning(self, "Error", f"No se pudo leer el frame {frame_number} ni el recovery {recovery}. Cerrando.")
                    self.close()
                    return
                else:
                    frame_number = recovery
                    if not self.revision_mode:
                        self.current_frame = frame_number
            else:
                QMessageBox.warning(self, "Error lectura", f"No se pudo leer el frame {frame_number}.")
                return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.shape[1]*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(900, 600, Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)
        self.label_info.setText(f"{self.video_name} - Frame {frame_number}/{self.total_frames - 1 if self.total_frames>0 else 0}")
        self.progress.setMaximum(max(self.total_frames - 1, 0))
        self.progress.setValue(frame_number)

        # cargar etiquetas previas si existen
        prev = etiquetas_df[(etiquetas_df["video"] == self.video_name) & (etiquetas_df["frame"] == frame_number)]
        if len(prev) > 0:
            prev_row = prev.iloc[-1]
            for i, err in enumerate(ERRORES):
                self.checkboxes[i].setChecked(bool(prev_row[err]))
        else:
            for cb in self.checkboxes:
                cb.setChecked(False)

    def actualizar_etiquetas(self):
        for i, cb in enumerate(self.checkboxes):
            self.etiquetas[i] = 1 if cb.isChecked() else 0

    def next_frame(self):
        global etiquetas_df
        if not self.revision_mode:
            frame_idx = self.current_frame
            # clamp
            if frame_idx < 0:
                frame_idx = 0
            if frame_idx >= self.total_frames:
                frame_idx = self.total_frames - 1

            # eliminar fila previa (si existe) para evitar duplicados
            etiquetas_df = etiquetas_df[~((etiquetas_df["video"] == self.video_name) & (etiquetas_df["frame"] == frame_idx))]

            # nueva fila
            nueva_fila = {"video": self.video_name, "frame": frame_idx}
            for i, err in enumerate(ERRORES):
                nueva_fila[err] = self.etiquetas[i]
            etiquetas_df = pd.concat([etiquetas_df, pd.DataFrame([nueva_fila])], ignore_index=True)

            # guardado periódico
            if frame_idx % 20 == 0:
                guardar_csv_ordenado()

            # avanzar (clamp)
            if frame_idx >= self.total_frames - 1:
                # llegamos al final del video
                QMessageBox.information(self, "Fin video", f"Has llegado al final del video {self.video_name}.")
                # cambiar a modo revisión automáticamente si hay etiquetas
                frames_existentes = etiquetas_df.loc[
                    (etiquetas_df["video"] == self.video_name) & (etiquetas_df[ERRORES].sum(axis=1) > 0), "frame"
                ].tolist()
                if len(frames_existentes) > 0:
                    self.revision_mode = True
                    self.frames_revisar = sorted(frames_existentes)
                    self.current_frame = 0
                else:
                    # cerrar o avanzar a siguiente video
                    self.close()
                    return
            else:
                self.current_frame = frame_idx + 1
        else:
            # modo revisión: avanzar en la lista de frames_revisar
            if self.current_frame < len(self.frames_revisar) - 1:
                self.current_frame += 1

        self.update_frame()

    def prev_frame(self):
        # retroceder en cualquiera de los modos (si estamos en el primer frame, no hacemos nada)
        if self.revision_mode:
            if self.current_frame > 0:
                self.current_frame -= 1
        else:
            if self.current_frame > 0:
                self.current_frame -= 1
        self.update_frame()

    def save_and_exit(self):
        guardar_csv_ordenado()
        QMessageBox.information(self, "Guardado", "💾 Progreso guardado correctamente.")
        self.close()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Right or key == Qt.Key_Space:
            self.next_frame()
        elif key == Qt.Key_Left:
            self.prev_frame()
        elif key == Qt.Key_S:
            self.save_and_exit()
        elif Qt.Key_1 <= key <= Qt.Key_6:
            idx = key - Qt.Key_1
            # toggle checkbox
            self.checkboxes[idx].setChecked(not self.checkboxes[idx].isChecked())


# ==== MENÚ PRINCIPAL ====
class MenuPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📂 Selector de Videos")
        self.setGeometry(300, 200, 600, 400)

        layout = QVBoxLayout()
        self.label = QLabel("Selecciona un video:")
        layout.addWidget(self.label)

        self.video_list = QListWidget()
        self.video_paths = VIDEO_PATHS  # usamos la lista global ordenada
        for v in self.video_paths:
            self.video_list.addItem(os.path.basename(v))
        layout.addWidget(self.video_list)

        btn_layout = QHBoxLayout()
        self.btn_etiquetar = QPushButton("✏️ Etiquetar")
        self.btn_etiquetar.clicked.connect(self.etiquetar)
        btn_layout.addWidget(self.btn_etiquetar)

        self.btn_revisar = QPushButton("👁️ Revisar etiquetas")
        self.btn_revisar.clicked.connect(self.revisar)
        btn_layout.addWidget(self.btn_revisar)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def etiquetar(self):
        idx = self.video_list.currentRow()
        if idx >= 0:
            path = self.video_paths[idx]
            self.etiquetador = Etiquetador(path)
            self.etiquetador.show()

    def revisar(self):
        idx = self.video_list.currentRow()
        if idx >= 0:
            path = self.video_paths[idx]
            self.etiquetador = Etiquetador(path, revision_mode=True)
            self.etiquetador.show()


# ==== MAIN ====
if __name__ == "__main__":
    app = QApplication([])
    window = MenuPrincipal()
    window.show()
    app.exec_()
