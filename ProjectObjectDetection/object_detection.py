import cv2
import numpy as np
import os


model_path = os.path.abspath("mobilenet_iter_73000.caffemodel") 
prototxt_path = os.path.abspath("deploy.prototxt")  


if not os.path.exists(model_path):
    print(f"Error: The model file {model_path} does not exist.")
    exit()

if not os.path.exists(prototxt_path):
    print(f"Error: The prototxt file {prototxt_path} does not exist.")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]


input_image_path = os.path.abspath("/home/xfin/ProjectObjectDetection/images/input.jpg") 
image = cv2.imread(input_image_path)

if image is None:
    print(f"Error: Could not load image at path: {input_image_path}")
    exit()

(h, w) = image.shape[:2]


blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


net.setInput(blob)


detections = net.forward()

# image detections and process
for i in range(detections.shape[2]):
    
    confidence = detections[0, 0, i, 2]

    
    if confidence > 0.2:  
        
        idx = int(detections[0, 0, i, 1])

        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        
        label = f"{CLASSES[idx]}: {confidence:.2f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


output_image_path = "output.jpg"
cv2.imwrite(output_image_path, image)
print(f"Output image saved at {output_image_path}")

