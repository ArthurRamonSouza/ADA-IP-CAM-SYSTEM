import cv2
import numpy as np

# Model path
model_path = "mobilenet_iter_73000.caffemodel"
prototxt_path = "deploy.prototxt"

# Loading the MobileNet (SSD) model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Setting classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initializing people tracking variables
entrances = 0
exits = 0
line_position = 200
trackers = []

def detect_movement_up_down(y, prev_y, line_position):
    if prev_y < line_position <= y:  # Down movement
        return "entrance"
    elif prev_y > line_position >= y:  # Up moviment
        return "exit"
    return None

previous_positions = {}

def detect_movement_left_right(current_x, prev_x, line_position):
    if prev_x < line_position <= current_x:
        return "entrance"
    elif prev_x > line_position >= current_x:
        return "exit"
    return None

# Variables for RTSP connection
user = 'admin'
password = '123456'
port = '554'
local_ip_address = '192.168.18.71'
resource = 'live'

# Extra variable for HTTP connection
router_ip_address = '192.168.18.1'

# RTSP URL formed
rtsp_url = f"rtsp://{user}:{password}@{local_ip_address}:{port}/{resource}"

# HTTP URL formed
http_url = f"http://{router_ip_address}:{port}"

# Creating the capture using rtsp ptorcol and FFMPEG decoder
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Main loop
while True:
    # If capture fail, init capture again
    if not cap.isOpened():
        print("Não foi possível abrir o stream RTSP.")
        # Creating the capture using rtsp ptorcol and FFMPEG decoder
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    else:
        # Starting capture
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            cap.release()

        else:

            # Resizing the frame to throw in the model
            (h, w) = frame.shape[:2]
            line_position = w // 2
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Processing the image
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.4:  # Threshold de confiança
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] == "person":  # Filtrar para apenas pessoas
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        person_id = i  # Usando índice como identificador simples

                        if person_id not in previous_positions:
                            previous_positions[person_id] = startX  # Inicializa `prev_x`

                        # # Adicionar rastreamento de posição Up and Down
                        # prev_y = startY  # Salvar a posição Y anterior para detectar movimento
                        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        # text = "Person: {:.2f}".format(confidence)
                        # cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                       # Detecta movimento
                        movement = detect_movement_left_right(startX, previous_positions[person_id], line_position)
                        if movement == "entrance":
                            entrances += 1
                        elif movement == "exit":
                            exits += 1

                        # Atualiza posição anterior
                        previous_positions[person_id] = startX

                        # Desenho do retângulo e exibição
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        text = "Person ({}): {:.2f}".format(person_id, confidence)
                        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Drawnig the line
            cv2.line(frame, (line_position, 0), (line_position, h), (0, 0, 255), 2)
            cv2.putText(frame, f"Entradas: {entrances}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Saidas: {exits}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


            # Open a window with the video
            cv2.imshow("Stream da Câmera", frame)
        
            # To exit press Q, for some reason clicking on the close window button another window will appear 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
