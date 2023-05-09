import cv2
import numpy as np

# load the YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# starting the web camera
cap = cv2.VideoCapture('video.mp4')

min_width_rectangle = 80
min_height_rectangle = 80
count_line_position = 550
Side_margin = 630

# initialize substractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    channelX = x + x1
    channelY = y + y1

    return channelX, channelY


def detect_vehicle_classes(frame, net, output_layers):
    # Get the height and width of the input frame
    height, width, channels = frame.shape

    # Create a blob from the input frame and set it as the input for the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (216, 216), (0, 0, 0), True, crop=True)  # DNN :- deep neural network
    net.setInput(blob)

    # Get the network outputs
    outs = net.forward(output_layers)

    # Initialize the lists to store the detected class IDs and confidences
    class_ids = []
    confidences = []

    # Loop over the detections and extract the class ID and confidence if it is a vehicle

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                if class_id == 2 or class_id == 5 or class_id == 7 or class_id == 3 or class_id == 1:  # class 2 - car , class 3 - motorcycle , class 5 - Bus , classs 7 - Truck
                    class_ids.append(class_id)
                    confidences.append(float(confidence))


                    # #Display the confidence score on top of the box
                    text = f'{int(confidence * 100)}%'


    return class_ids, confidences


detector = []
offset = 6  # allowable error in pixel
counter = 0
Incoming = 0
Outgoing = 0

vehicle_ids = []

while (True):
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # applying on its all frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    # cv2.line(frame1,(Side_margin,count_line_position),(Side_margin,count_line_position+50),(255,127,0),3)

    for (i, c) in enumerate(counterShape):
        (x, y, weidth, height) = cv2.boundingRect(c)
        val_counter = (weidth >= min_width_rectangle) and (height >= min_height_rectangle)
        if not val_counter:
            continue

        # to make the rectangle faround our bgsem (OpenCV Box)
        cv2.rectangle(frame1, (x, y), (x + weidth, y + height), (0, 255, 0), 2)

        center = center_handle(x, y, weidth, height)
        detector.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        # Classify the vehicle in the detected region
        if len(detector) > 0:
            class_ids, confidences = detect_vehicle_classes(frame1[y:y + height, x:x + weidth], net, output_layers)

            # Print the detected classes and their confidences
            for i in range(len(class_ids)):
                print(f"Detected vehicle class: {class_ids[i]}, Confidence: {confidences[i]}")

        for (x, y) in detector:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                if x > Side_margin:
                    Outgoing += 1
                else:
                    Incoming += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (127, 255), 3)
                detector.remove((x, y))
                print("Vehicle Counter  : " + str(counter))
                print("Vehicle Coming   : " + str(Incoming))
                print("Vehicle Outgoing : " + str(Outgoing))

    cv2.putText(frame1, "VEHICLE COUNTER : " + str(counter), (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    cv2.putText(frame1, "VEHICLE COMING : " + str(Incoming), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame1, "VEHICLE OUTGOING : " + str(Outgoing), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release
