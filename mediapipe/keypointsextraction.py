# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 23:27:19 2025

@author: user
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Ruta del modelo y del video
model_path = 'D:/la-u/ciclo 2025-1/Seminario/MODELO/Keypoints/mediaPipe/pose_landmarker_heavy.task'
video_path = 'D:/la-u/ciclo 2025-1/Seminario/DATASET/Diego_Romero/Diego_Romero1.mp4'
output_path = 'output_pose.mp4'  # salida del video

# Configurar MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

landmarker = PoseLandmarker.create_from_options(options)

# Abrir video
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Obtener tamaño del video original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar el VideoWriter para salida mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((frame_count / fps) * 1000)

    # Detectar poses
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.pose_landmarks:
        for pose in results.pose_landmarks:
            for landmark in pose:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Escribir el frame con puntos en el video de salida
    out.write(frame)

    # Mostrar opcionalmente
    cv2.imshow('Detección de Pose', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()