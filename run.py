import cv2
import onnxruntime as ort
import numpy as np
import time

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best.onnx"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.3


CLASS_NAMES = ["pothole", "manhole", "crack", "patch", "other1", "other2", "other3"]


session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera started...")
print("Press Q to exit")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w, _ = frame.shape

    
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0

    
    outputs = session.run(None, {input_name: img})
    predictions = np.squeeze(outputs[0]).T

    
    for pred in predictions:
        confidence = np.max(pred[4:])
        class_id = np.argmax(pred[4:])

        if confidence > CONF_THRESHOLD:
            x_center, y_center, w, h = pred[:4]

            
            x1 = int((x_center - w / 2) * orig_w / INPUT_SIZE)
            y1 = int((y_center - h / 2) * orig_h / INPUT_SIZE)
            x2 = int((x_center + w / 2) * orig_w / INPUT_SIZE)
            y2 = int((y_center + h / 2) * orig_h / INPUT_SIZE)

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2)

    
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow("Road Anomaly Detection - 320", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()  
