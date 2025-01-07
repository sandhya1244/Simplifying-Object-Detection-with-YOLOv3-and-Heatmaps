import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load YOLO model
model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layers
layer_names = model.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[int(i[0]) - 1] for i in model.getUnconnectedOutLayers()]

# Video input and output
cap = cv2.VideoCapture('13.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize the VideoWriter with the correct frame size
output_video = cv2.VideoWriter('output_video.mp4', 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               30, 
                               (frame_width, frame_height))  # Ensure dimensions match the frame

# Heatmap initialization
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

# Process video frames
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# Matplotlib figure setup (move it outside the loop)
plt.ion()
fig, ax = plt.subplots()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outputs = model.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Parse YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and update heatmap
    frame_heatmap = np.zeros_like(heatmap)  # Frame-specific heatmap
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i  # Safely handle scalar or list
        box = boxes[i]
        x, y, w, h = box
        frame_heatmap[y:y+h, x:x+w] += 1
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Normalize and overlay heatmap
    heatmap_normalized = cv2.normalize(frame_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    frame_with_heatmap = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

    # Count objects
    counts = {classes[i]: 0 for i in set(class_ids)}
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i  # Safely handle scalar or list
        label = str(classes[class_ids[i]])
        counts[label] += 1

    # Display counts on the frame
    y_offset = 20
    for label, count in counts.items():
        cv2.putText(frame_with_heatmap, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20

    # Write frame to output video
    output_video.write(frame_with_heatmap)

    # Display frame using matplotlib (avoid blocking)
    ax.clear()  # Clear previous frame
    ax.imshow(cv2.cvtColor(frame_with_heatmap, cv2.COLOR_BGR2RGB))
    ax.axis('off')  # Hide axis
    plt.draw()
    plt.pause(0.1)  # Pause to update the display

    pbar.update(1)

# Release resources
cap.release()
output_video.release()
pbar.close()
plt.ioff()  # Turn off interactive mode
plt.close(fig)
cv2.destroyAllWindows()

print("Processing complete. Output saved as 'output_video.mp4'.")
