import os
import cv2
import mediapipe as mp

def show_video_with_keypoints(
    video_path: str,
    model_path: str
):
    """
    Reproduce el vídeo frame a frame, detecta keypoints en modo VIDEO
    y los dibuja en pantalla.
    """
    # Carga vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir: {video_path}")

    # Inicializar MediaPipe PoseLandmarker en modo VIDEO
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    landmarker = PoseLandmarker.create_from_options(options)

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR → RGB para MediaPipe
        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        timestamp_ms = int((frame_count / fps) * 1000)

        # Detección con tracking
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Dibujar keypoints
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for pose in results.pose_landmarks:
                for lm in pose:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)

        # Mostrar
        cv2.imshow('PoseLandmarker VIDEO Mode', frame)
        frame_count += 1

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "D:/la-u/ciclo 2025-1/Seminario/DATASET/Hugo/20250510_145932.mp4"
    model_path = "D:/la-u/ciclo 2025-1/Seminario/MODELO/Keypoints/mediaPipe/pose_landmarker_heavy.task"

    show_video_with_keypoints(video_path, model_path)
