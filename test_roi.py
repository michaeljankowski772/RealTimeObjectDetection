import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#model.to("cuda")

with open("config.json") as f:
    config = json.load(f)

caps = {}

for cam in config["cameras"]:
    if cam["enabled"]:
        caps[cam["name"]] = cv2.VideoCapture(cam["url"], cv2.CAP_FFMPEG)


frame_count = 0

while True:
    ret, frame = caps["test1"].read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 != 0:
        continue 

    h, w = frame.shape[:2]

    #frame_small = cv2.resize(frame, (640, 640))
    #h_s, w_s = frame_small.shape[:2]

    roi = frame[h//2:h, 0:w]

    roi_small = cv2.resize(roi, (640, 640))
    results = model(roi_small) #classes=[0])  #  person
    #allowed = ["person", "car"]

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            #if label in allowed and confidence > 0.3:
            if confidence > 0.3:
                print(f"{label} {confidence:.2f}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])


                scale_x = w / 640
                scale_y = (h/2) / 640

                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y + h//2)  # ROI
                y2 = int(y2 * scale_y + h//2)


                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.5, (0,0,255), 8)

    scale_display = 0.7
    frame_display = cv2.resize(frame, (int(w*scale_display), int(h*scale_display)))

    cv2.imshow("kamera", frame_display)

    if cv2.waitKey(1) == 27:
        break

caps["test1"].release()
cv2.destroyAllWindows()