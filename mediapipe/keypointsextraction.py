# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 23:27:19 2025

@author: user
"""
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'D:/la-u/ciclo 2025-1/TPI/MODELO/Keypoints/mediaPipe/pose_landmarker_heavy.task'


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)


landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture('D:/la-u/ciclo 2025-1/TPI/MODELO/Video pruba/luis2.mp4')
frame_count = 0


csv_file = open('keypoints_output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'landmark_index', 'x', 'y', 'z', 'visibility'])


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    timestamp_ms = int((frame_count / fps) * 1000)
    
    # Detectar los keypoints en el frame actual
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if results.pose_landmarks:
        for pose in results.pose_landmarks:
            for landmark in pose:
                # Convertir coordenadas normalizadas a pixeles
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        for pose_index, pose in enumerate(results.pose_landmarks):
            for idx, landmark in enumerate(pose):
                csv_writer.writerow([
                    frame_count,  # el número de frame
                    idx,          # índice del keypoint (0-32 para pose)
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                    ])


    # Mostrar el frame procesado
    cv2.imshow('Detección de Pose', frame)
    frame_count += 1

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()


