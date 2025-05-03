import os
import csv
import cv2
import mediapipe as mp
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

# ============================================================
# FUNCIONES
# ============================================================

def init_landmarker(model_path: str):
    """ Inicializa el PoseLandmarker (por proceso). """
    global landmarker
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    landmarker = PoseLandmarker.create_from_options(options)

def process_video_worker(args: Tuple[str, str]) -> List[List]:
    """ Worker para multiprocessing. """
    video_path, model_path = args
    init_landmarker(model_path)  # Inicializa landmarker en este proceso
    return process_video(video_path)

def process_video(video_path: str) -> List[List]:
    """
    Procesa un único vídeo y devuelve una lista de filas:
      [video_filename (con extensión), frame_idx, landmark_idx, x, y, z, visibility]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ No se pudo abrir {video_path}")
        return []

    rows = []
    frame_count = 0
    video_filename = os.path.basename(video_path)  # Con extensión

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR → RGB y crear mp.Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detectar pose
        try:
            results = landmarker.detect(mp_image)
        except Exception as e:
            print(f"❗ Error en frame {frame_count} de {video_filename}: {e}")
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
    video_extensions: tuple = ('.mp4', '.avi', '.mov')
):
    """ Procesa todos los videos en paralelo y genera un único CSV. """
    # Recopilar todos los videos
    video_paths = []
    for dirpath, _, files in os.walk(root_videos_dir):
        for fname in files:
            if fname.lower().endswith(video_extensions):
                video_path = os.path.join(dirpath, fname)
                video_paths.append(video_path)

    print(f"🔍 {len(video_paths)} videos encontrados. Usando {cpu_count()} núcleos...")

    # Crear pool de workers
    with Pool(processes=cpu_count()) as pool:
        all_rows_lists = pool.map(
            func=process_video_worker,
            iterable=[(vp, model_path) for vp in video_paths]
        )

    # Aplanar resultados
    all_rows = [row for rows in all_rows_lists for row in rows]

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Escribir CSV
    with open(output_csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'video_filename',
            'frame',
            'landmark_index',
            'x', 'y', 'z',
            'visibility'
        ])
        writer.writerows(all_rows)

    print(f"🎉 CSV generado en: {output_csv_path}")

# ============================================================
# USO
# ============================================================

if __name__ == "__main__":
    root_videos_dir = "D:/la-u/ciclo 2025-1/Seminario/DATASETPRUEBA"
    model_path       = "D:/la-u/ciclo 2025-1/Seminario/MODELO/Keypoints/mediaPipe/pose_landmarker_lite.task"
    output_csv_path  = "D:/la-u/ciclo 2025-1/Seminario/Proyecto/mediapipe/csv/kp_dataset_mp.csv"

    generate_keypoints_csv_parallel(root_videos_dir, model_path, output_csv_path)