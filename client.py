import socket
import cv2 as cv
import struct
import numpy as np
import pickle

s = socket.socket()
cap = cv.VideoCapture(0)

HOST = '192.168.2.50'
PORT = 8800

try:
	s.connect((HOST,PORT))
	while True:
		_,frame = cap.read()
		_,frame_buffer = cv.imencode(".jpg",frame)
		frame_bytes = frame_buffer.tobytes()
		
		frame_size = len(frame_bytes)
		
		s.sendall(struct.pack('!I',frame_size))
		s.sendall(frame_bytes)
		
		feedback = s.recv(1024)
		if feedback == "ACK":
			continue
		
		processed = b""
		
		processed_size = s.recv(4)
		res_size = struct.unpack('!I',processed_size)[0]
		while len(processed) < res_size:
			chunk = s.recv(4096)
			if not chunk:
				break
			processed += chunk
		
		processed_array = np.frombuffer(processed,dtype=np.uint8)
		res = cv.imdecode(processed_array,flags = cv.IMREAD_COLOR)
		cv.imshow("RECEIVED RASPI",res)
		cv.waitKey(1)
		
except socket.error as msg:
	print("Socket Error", str(msg))
finally:
	cap.release()
	s.close()
