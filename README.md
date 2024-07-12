# AUTOMATIC-AMBULANCE-DETECTION-SYSTEM-FOR-EMERGENCY-VEHICLE-PRIORITIZATION-USING-IMAGE-PROCESSING
      
This project focuses on developing an automatic ambulance detection system using YOLO (You Only Look Once) image processing to prioritize emergency vehicles in traffic. The system detects ambulances in real-time through camera feeds, enabling traffic lights to give them the right-of-way immediately. By automating the detection and prioritization process, it reduces delays for emergency vehicles, ensuring faster response times. The system's integration into existing traffic management infrastructure enhances overall traffic flow and safety. This innovative approach contributes to more efficient urban traffic management and better emergency services.

# FEATURES USED:
1. Real-time ambulance detection
2. Traffic signal control

# REQUIREMENTS :
1. PYTHON
2. Required Libraries :
     a. YOLO from Ultralytics
     b. Open Source Computer Vision
     c. Tkinter

# ARCHITECTURE DIAGRAM :
![image](https://github.com/S-ABHISHEK-1905/AUTOMATIC-AMBULANCE-DETECTION-SYSTEM-FOR-EMERGENCY-VEHICLE-PRIORITIZATION-USING-IMAGE-PROCESSING/assets/66360846/a12dccb5-79f5-4878-a59b-0983f019e7a0)

# FLOW DIAGRAM :
![FLOW](https://github.com/S-ABHISHEK-1905/AUTOMATIC-AMBULANCE-DETECTION-SYSTEM-FOR-EMERGENCY-VEHICLE-PRIORITIZATION-USING-IMAGE-PROCESSING/assets/66360846/96c4c4cd-a076-473c-a3e6-d979d07b3aed)

# INSTALLATIONS
### INSTALL THE REQUIRED PACKAGES
```
    from ultralytics import YOLO

    import cv2

    import tkinter as tk
```

### DOWNLOADING THE IMAGE DATASET
    We use Open image dataset for downloading image datasets of Ambulance

### ANNOTATE THE IMAGES IN DATASET
    For annotating the images for training the dataset we use Roboflow .

# USAGE
### 1. Collect the dataset .
### 2. Annotate the images .
### 3. Train the images using YOLO .
### 4. Detect the ambulance in the given video .
### 5. Show a Green signal after the Ambulance is detected .
### 6. End of the Program .


# PROGRAM
## 1. YOLO CODE TO TRAIN IMAGE DATASET

```

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.yaml")

results = model.train(data="D:/code/data.yaml", epochs=30)

```

## 2. CODE TO PREDICT AMBULANCE IN VIDEO

```

import cv2
from ultralytics import YOLO
import tkinter as tk

# Define the input video path or use a webcam (0)
video_path = "D:/PAA/videos/ambu1.mp4"  # Replace with your video path or set to 0 for webcam

# Load the trained YOLO model
model = YOLO("D:/PAA/runs/detect/train3/weights/best.pt")  # Replace with your model path

# Define the threshold for object detection
threshold = 0.5

window = tk.Tk()
window.title("Ambulance Detection Traffic Light")

# Create the traffic light frame
traffic_light_frame = tk.Frame(window, bg="green", width=100, height=100)
traffic_light_frame.pack()


def detect_ambulance():
    global light_on

    # Simulate ambulance detection by changing the light color to green
    light_on = "green"
    traffic_light_frame.config(bg=light_on)


# Open the video stream or capture from a live camera
if video_path:
    cap = cv2.VideoCapture(video_path)  # For video file
else:
    cap = cv2.VideoCapture(0)  # For live camera

while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    # Detect objects in the frame using YOLO
    results = model(frame)[0]

    # Draw bounding boxes and labels around detected objects
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and class_id == 0:  # Check if class_id corresponds to ambulance
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, "Ambulance Detected", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            window.after(5000, detect_ambulance)
            window.mainloop()

    # Display the resulting frame with detected objects
    cv2.imshow("Ambulance Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

```


# OUTPUT
## AMBULANCE DETECTED
![image](https://github.com/S-ABHISHEK-1905/AUTOMATIC-AMBULANCE-DETECTION-SYSTEM-FOR-EMERGENCY-VEHICLE-PRIORITIZATION-USING-IMAGE-PROCESSING/assets/66360846/a14269f3-dd56-4640-99d2-5f84da618fd3)


## TRAFFIC LIGHT : GREEN ( AFTER DETECTED )
![image](https://github.com/S-ABHISHEK-1905/AUTOMATIC-AMBULANCE-DETECTION-SYSTEM-FOR-EMERGENCY-VEHICLE-PRIORITIZATION-USING-IMAGE-PROCESSING/assets/66360846/cc6fd5d6-14f5-4f2e-9765-b609be0cc036)


# RESULT
1. An automatic ambulance detection system using image processing techniques has been developed to enhance emergency response times and improve traffic management efficiency.

2. The system accurately identifies and localizes ambulances in real-time video streams using a combination of background subtraction, object segmentation, feature extraction, and machine learning techniques.

3. The successful implementation of this system holds the potential to revolutionize emergency response protocols and save lives.

    
