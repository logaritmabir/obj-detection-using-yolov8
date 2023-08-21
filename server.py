import socket
import cv2 as cv
import numpy as np
import struct
from ultralytics import YOLO
import math
import time
import cvzone


def measure_distance(x):
    return 41.8 + (-2.9e-03 * x) + (6.49e-08 * x ** 2)
class_names = ['flash']

model = YOLO("blured.pt")

rect_top_left_x = 350
rect_top_left_y = 300

rect_bottom_right_x = 500
rect_bottom_right_y = 360

center_x = int((rect_top_left_x + rect_bottom_right_x)/2)
center_y = int((rect_bottom_right_y+rect_top_left_y)/2)

center_of_rect = (center_x,center_y)
rect_color = (0,0,255)

HOST = '192.168.2.50'
PORT = 8800

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket has been initialized")

    s.bind((HOST, PORT))
    print(f"Socket has been connected to port: {PORT}")

    s.listen(1)
    print("Socket is listening for 1 device")
except socket.error as msg:
    print("An error occurred during socket creation:", str(msg))

client_socket, client_adr = s.accept()
print(f"Connection has been established between HOST and {client_adr}")

try:
    while True:
        frame_size = client_socket.recv(4)
        data_size = struct.unpack('!I', frame_size)[0]
        print(data_size)
        data = b""
        while len(data) < data_size :
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            data += chunk
        
        ack_message = "ACK".encode('utf-8')
        client_socket.sendall(ack_message)

        frame_array = np.frombuffer(data,dtype=np.uint8)
        frame = cv.imdecode(frame_array, flags=cv.IMREAD_COLOR)
        frame_capture_time = time.time()
        cv.imshow("received frame",frame)
        cv.rectangle(frame,(rect_top_left_x,rect_top_left_y),(rect_bottom_right_x,rect_bottom_right_y),rect_color,3)
        cv.circle(frame,center_of_rect,3,(255,0,0),-1)

        result = model(frame)

        for res in result:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detected_center_x = int((x1+x2)/2)
                detected_center_y = int((y1+y2)/2)
                center_of_detected = (detected_center_x, detected_center_y)

                cv.circle(frame,center_of_detected,3,(255,0,0),-1)
                cv.line(frame,center_of_rect,center_of_detected,(0,255,255),1)

                line_length = math.sqrt((detected_center_x - center_x)**2 + (detected_center_y - center_y)**2)
                print(line_length)
                
                if line_length < 20:
                    rect_color = (0,255,0)
                else:
                    rect_color = (0,0,255)
                
                w,h = abs(x2-x1),abs(y2-y1)
                area = w * h
                dist =  (measure_distance(area) * 100) / 100

                cvzone.cornerRect(frame,(x1,y1,w,h))
                cls = int(box.cls[0])
                cvzone.putTextRect(frame,f"{class_names[cls]} {math.ceil(box.conf[0]*100)/100} dist : {dist}",(max(0, x1), max(35, y1)), scale=1, thickness=1)
                # detected_obj = frame[y1:y2, x1:x2]
                # cv.imshow("roi",detected_obj)

        end = time.time()
        fps = 1 / (end - frame_capture_time)
        print(f"fps : {fps}")
        cv.imshow("frame",frame)

        frame_buffer = cv.imencode(".jpg",frame)[1]
        frame_bytes = frame_buffer.tobytes()

        frame_size = len(frame_bytes)
        client_socket.sendall(struct.pack('!L',frame_size))
        client_socket.sendall(frame_bytes)
        cv.waitKey(1)

except KeyboardInterrupt:
    print("Server stopped by the user.")
finally:
    s.close()
    client_socket.close()