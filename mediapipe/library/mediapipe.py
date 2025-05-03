import os
import csv
import cv2
import mediapipe as mp
from typing import List

def process_video(
    video_path: str,
    landmarker: mp.tasks.vision.PoseLandmarker
) -> List[List]:
    """
    Procesa un único vídeo y devuelve una lista de filas:
      [video_filename (sin extensión), frame_idx, landmark_idx, x, y, z, visibility]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ No se pudo abrir {video_path}")
        return []

    rows = []
    frame_count = 0
    video_filename = os.path.splitext(os.path.basename(video_path))[0]  # <- Sin extensión

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

        # Detectar pose (modo IMAGE para no lidiar con timestamps)
        try:
            results = landmarker.detect(mp_image)
        except Exception as e:
            print(f"❗ Error en frame {frame_count} de {video_filename}: {e}")
            frame_count += 1
            continue

        # Extraer y acumular keypoints
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


def generate_keypoints_csv(
    root_videos_dir: str,
    model_path: str,
    output_csv_path: str,
    video_extensions: tuple = ('.mp4', '.avi', '.mov')
):
    """
    Recorre todas las subcarpetas de root_videos_dir, procesa cada vídeo
    con process_video y escribe todos los keypoints en output_csv_path.
    """
    # Inicializar modelo una sola vez
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # Preparar CSV
    with open(output_csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'video_filename',
            'frame',
            'landmark_index',
            'x', 'y', 'z',
            'visibility'
        ])

        # Iterar sobre todos los vídeos
        for dirpath, _, files in os.walk(root_videos_dir):
            for fname in files:
                if not fname.lower().endswith(video_extensions):
                    continue

                video_path = os.path.join(dirpath, fname)
                rows = process_video(video_path, landmarker)
                writer.writerows(rows)

    print(f"🎉 CSV generado en: {output_csv_path}")
