from pathlib import Path 
import time 
import cv2

motion_frame = Path("assets/aula8_video_exemplo.avi")

capture = cv2.VideoCapture(str(motion_frame))

if not capture.isOpened():
    print(f"Falaha ao abrir o arquivo de vídeo: {motion_frame}")
    exit()

fps = capture.get(cv2.CAP_PROP_FPS)
period = 1/ fps if fps > 0 else 1/30 

previus_frame = None

while True: 
    start = time.perf_counter()
    ret, frame = capture.read() 

    if not ret:
        print("Falha ao ler o frame do vídeo.")
        continue  

    gaussian_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_frame = cv2.GaussianBlur(gaussian_frame, (15, 15), 0)

    if previus_frame is None:
        previus_frame = gaussian_frame
        continue

    ditinct_frame = cv2.absdiff(previus_frame, gaussian_frame)

    _, moviment_frame = cv2.threshold(ditinct_frame, 12, 255, cv2.THRESH_BINARY)

    moviment_frames = cv2.dilate(moviment_frame, None, iterations=2) 

    contours, _ = cv2.findContours(
        moviment_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        
        x, y, largura, altura = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)


    cv2.imshow("Analise dos frames", frame) 

    previus_frame = gaussian_frame

    time_cicle = time.perf_counter() - start
    time_wait_ms = max(1, int((period - time_cicle) * 1000))

    if cv2.waitKey(time_wait_ms) & 0xFF == ord("e"):
        break 

capture.release()
cv2.destroyAllWindows()
