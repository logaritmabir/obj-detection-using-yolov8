# obj-detection-using-yolov8
training yolo network on custom dataset


train.py -> Detecting custom object via custom .pt file then detecting the objects in the aspect of monocular camera. Measure distance and tracking in addition. (~20 fps)\

server.py and client.py -> Raspberry's processor is not good enough to run .pt file (even if it is quad-core, 0.9 fps). An external computer has been used to process data (server).
Raspberry (client) captures frame from camera module then sends the data to server. Server runs the .pt file on the image then sends back the output to client. (~11 fps)

## Proposed protocol between of Raspberry and Server

Step 1 : Client captures data from camera module.\
Step 2 : Client packages (serialization) the data then sends the size of package and the data respectively.\
Step 3 : Client starts to wait for the process (Waits for the "ACK" message that means server successfully received the package).\
Step 4 : Server receives the data size.\
Step 5 : Server receives packages until it reaches the expected data size.\
Step 6 : Server runs the trained model on the frame.\
Step 7 : Server packages the processed frame and sends size of the package and the data respectively.
