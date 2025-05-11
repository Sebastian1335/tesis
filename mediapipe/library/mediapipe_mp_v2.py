import os

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'  # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# Ahora importa tus librerías de forma "limpia":
import csv
import cv2
import mediapipe as mp
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

def process_video(args: Tuple[str, str]) -> List[List]:
    """
    Procesa un vídeo en modo VIDEO, creando su propio landmarker,
    y devuelve filas [video_filename, frame_idx, landmark_idx, x,y,z,visibility].
    """
    video_path, model_path = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ No se pudo abrir {video_path}")
        return []

    # --- Inicializar landmarker en modo VIDEO para este vídeo ---
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    landmarker = PoseLandmarker.create_from_options(options)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count  = 0
    rows         = []
    video_filename = os.path.basename(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preparar imagen
        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((frame_count / fps) * 1000)

        # Detectar con tracking
        try:
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"❗ Error frame {frame_count} de {video_filename}: {e}")
            frame_count += 1
            continue

        # Extraer keypoints
        if results.pose_landmarks:
            for pose in results.pose_landmarks:
                for idx, lm in enumerate(pose):
                    rows.append([
                        video_filename,
                        frame_count,
                        idx,
                        lm.x, lm.y, lm.z,
                        lm.visibility
                    ])

        frame_count += 1

    cap.release()
    print(f"✅ {video_filename}: {frame_count} frames procesados.")
    return rows


def generate_keypoints_csv_parallel(
    root_videos_dir: str,
    model_path: str,
    output_csv_path: str,
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')
):
    """
    Recorre todos los vídeos en paralelo (cada worker crea su landmarker VIDEO),
    y vuelca todos los keypoints al CSV.
    """
    # 1) Listar vídeos
    video_paths = []
    for dp, _, files in os.walk(root_videos_dir):
        for fn in files:
            if fn.lower().endswith(video_extensions):
                video_paths.append(os.path.join(dp, fn))

    print(f"🔍 {len(video_paths)} videos encontrados. Usando {cpu_count()} núcleos...")

    # 2) Procesar en paralelo
    with Pool(cpu_count()) as pool:
        all_rows_lists = pool.map(
            process_video,
            [(vp, model_path) for vp in video_paths]
        )

    # 3) Aplanar resultados
    all_rows = [row for sub in all_rows_lists for row in sub]

    # 4) Crear carpeta de salida si falta
    out_dir = os.path.dirname(output_csv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 5) Escribir CSV
    abs_path = os.path.abspath(output_csv_path)
    print(f"➡️ Escribiendo CSV en: {abs_path} ({len(all_rows)} filas)")
    with open(abs_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'video_filename', 'frame', 'landmark_index',
            'x', 'y', 'z', 'visibility'
        ])
        w.writerows(all_rows)

    print("🎉 CSV generado con éxito.")


if __name__ == "__main__":
    root_videos_dir = "D:/la-u/ciclo 2025-1/Seminario/DATASETPRUEBA"
    model_path      = "D:/la-u/ciclo 2025-1/Seminario/MODELO/Keypoints/mediaPipe/pose_landmarker_lite.task"
    output_csv_path = "./mediapipe/csv/kp_dataset_video_mode.csv"

    generate_keypoints_csv_parallel(
        root_videos_dir,
        model_path,
        output_csv_path
    )