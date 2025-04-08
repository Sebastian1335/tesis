# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 23:53:10 2025

@author: user
"""
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# Ruta al modelo
model_path = 'D:/la-u/ciclo 2025-1/TPI/MODELO/Keypoints/mediaPipe/pose_landmarker_heavy.task'

# Configuración de las opciones para el detector
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO  # Usamos VIDEO en lugar de LIVE
)

landmarker = PoseLandmarker.create_from_options(options)

# Abrir la webcam (generalmente el dispositivo 0)
cap = cv2.VideoCapture(0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir de BGR a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Calcular timestamp en milisegundos (si no se obtiene fps, se usa 30 por defecto)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    timestamp_ms = int((frame_count / fps) * 1000)

    # Procesar el frame para obtener los keypoints
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Dibujar los keypoints sobre el frame
    if result.pose_landmarks:
        for pose in result.pose_landmarks:
            for landmark in pose:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Pose en Vivo", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
