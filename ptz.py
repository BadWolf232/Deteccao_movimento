from pathlib import Path 
import time 
import cv2


motion_frame = Path("assets")

path_frames = sorted((motion_frame / "input").glob("in*.jpg"))

if not path_frames:
    print("Nenhum frame foi encontrado")
    exit()

start, end = map(int, (motion_frame/"temporalROI.txt").read_text().split())

roi = cv2.imread(str(motion_frame / "ROI.bmp"), cv2.IMREAD_GRAYSCALE)
_, roi_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)

previous_frame = None  

for path in path_frames: 

    frame_num = int(path.stem[2:])
    frame = cv2.imread(str(path))

    if frame is None:
        print(f"Falha ao le o frame: {path}")
        break
    

    gaussian_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_frame = cv2.GaussianBlur(gaussian_frame, (15, 15), 0)

    if previous_frame is None:
        previous_frame = gaussian_frame
        continue

    ditinct_frame = cv2.absdiff(previous_frame, gaussian_frame)
    _, moviment_frame = cv2.threshold(ditinct_frame, 12, 255, cv2.THRESH_BINARY)

    movement = cv2.bitwise_and(moviment_frame, roi_mask)
    movement = cv2.dilate(movement, None, iterations=2)

    contours, _ = cv2.findContours(
        movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        
        x, y, largura, altura = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)

    in_eval = start <= frame_num <= end 
    cv2.putText(
        frame, f"frase {frame_num} {'{avalicao}' if in_eval else ''}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )

    cv2.imshow("Analise dos frames", frame) 

    previous_frame = gaussian_frame

    #time_cicle = time.perf_counter() - start
    #time_wait_ms = max(1, int((period - time_cicle) * 1000))

    if cv2.waitKey(30) & 0xFF == ord("e"):
        break 

#capture.release()
cv2.destroyAllWindows()
