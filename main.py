import cv2

# Variables for RTSP connection
user = 'admin'
password = '123456'
port = '554'
ip_address = '192.168.18.71'
resource = 'live'

# RTSP URL formed
rtsp_url = f"rtsp://{user}:{password}@{ip_address}:{port}/{resource}"

# Creating the capture using rtsp ptorcol and FFMPEG decoder
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Main loop
while True:
    # If capture fail, init capture again
    if not cap.isOpened():
        print("Não foi possível abrir o stream RTSP.")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    else:
        # Starting capture
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            cap.release()

        else:
            # Open a window with the video
            cv2.imshow("Stream da Câmera", frame)
        
            # To exit press Q, for some reason clicking on the close window button another window will appear 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
