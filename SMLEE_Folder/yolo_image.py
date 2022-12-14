import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib as plt




# 점찍는 함수

white_canvas = np.zeros((1000, 1000, 3), dtype="uint8") + 255

def img_show(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


# Yolo 로드
net = cv2.dnn.readNet("../yolov3/yolov3.weights", "../yolov3/yolov3.cfg")

classes = []
with open("../yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



# 이미지 가져오기
img = cv2.imread("../image/people6.jpg")
# img = cv2.resize(img, None, fx=0.4, fy=0.4)

height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
center__x = []
center__y = []



for out in outs:

    for detection in out:

        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


            class_ids.append(class_id)




Kmean = KMeans(n_clusters=2)



red = (0,0,255)


countPeople = 0
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):

    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, "%s  (%d , %d)"%(label,x+w/2,y+h/2), (x, y + 30), font, 3, color, 3)
        img = cv2.line(img, (int(x+w/2), int(y+h/2)) , (int(x+w/2), int(y+h/2)), red , 5 )


        # white_canvas = cv2.line(black_canvas, (int(x+w/2), int(y+h/2)), (int(x+w/2), int(y+h/2)), red, 20 )
        white_canvas = cv2.line(white_canvas, (int(x+w/2), int(y+h/2)), (int(x+w/2), int(y+h/2)), red, 20 )
        print("x is ",x+w/2, "y is",y+h/2)
        countPeople += 1


print("counting: ", countPeople)


cv2.namedWindow('map', cv2.WINDOW_NORMAL)              # gray 이름으로 창 생성
cv2.imshow("Image", img)                                # origin 창에 이미지 표시
cv2.imshow('map', white_canvas)                            # map 창에 이미지 표시

cv2.moveWindow('map', 100, 100)                        # 창 위치 변경

cv2.waitKey(0)                                          # 아무키나 누르면
cv2.resizeWindow('map', 100, 100)                      # 창 크기 변경 (변경 됨))

cv2.waitKey(0)                                          # 아무키나 누르면
cv2.destroyWindow("map")                               # map 창 닫기


cv2.waitKey(0)
cv2.destroyAllWindows()
