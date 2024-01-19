import math


from kafka import KafkaConsumer
from datetime import datetime, timedelta
import pytz
import threading
import torch
import numpy as np
import cv2
import play_sound
import pymongo
import os

myclient = pymongo.MongoClient("mongodb://localhost:27017")
mydb = myclient["test"]
mycol = mydb["test_driver"]
model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/best (2).pt', force_reload=True)

alert_id = [0]

def yolo_dectect(model, frame,flag, mycol,alert_id):

    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Make detections
    results = model(new_frame)
    if len(results.xywh[0]) > 0:

        dclass = results.xywh[0][0][5]


        if dclass.item() == 16.0:
            flag[0] += 1
            if flag[0] > 20:
                result = mycol.find_one({"_id": alert_id[0]})
                if not result:
                    new_folder_path = "D:\\driver\\" + str(alert_id[0])
                    if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)
                        print(f"New folder created: {new_folder_path}")
                    else:
                        print(f"Folder already exists: {new_folder_path}")
                    print("-------------------------Khởi tạo ----------------------------------")
                    start_time = datetime.now().isoformat()
                    path_image = "driver_image_" + str(flag[0]) + ".jpg"
                    url = os.path.join(new_folder_path, path_image)
                    mydict = {"_id": alert_id[0],"driver_id":123, "driver_name": "Hoang Van Linh", "start_time": start_time, "driver_images": [url], "speed_images": []}
                    mycol.insert_one(mydict)
                    cv2.imwrite(url, frame)
                else:

                    new_path = "D:\\driver\\" + str(alert_id[0])
                    path_image = "driver_image_" + str(flag[0]) + ".jpg"
                    url = os.path.join( new_path, path_image)
                    document_query = {"_id": alert_id[0]}
                    mycol.update_one(document_query, {"$push": {"driver_images": url}})
                    cv2.imwrite(url, frame)

                # play_sound.play_sound()
        else:
            if (flag[0] > 20):
                print("========================================End===================================")
                end_time = datetime.now().isoformat()
                mycol.update_one({"_id": alert_id[0]}, {"$set": {"end_time": end_time}} )
                alert_id[0] +=1

            flag[0] = 0

        cv2.imshow('YOLO', np.squeeze(results.render()))

def night_speed(frame,mycol,alert_id):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, binary_frame = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
    _, binary_frame = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    tan_value = h / w
    angle_rad = math.atan(tan_value)
    angle_deg = math.degrees(angle_rad)
    speed = 0
    speed = 30 + angle_deg * 200/180
    cv2.putText(frame, f'speed: {speed}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    result = mycol.find_one({"_id": alert_id[0]})
    if result:
        print("Document found:")
        mycol.update_one({"_id": alert_id[0]}, {"$set": {"speed": speed}})
        new_folder_path = "D:\\speed\\" + str(alert_id[0])
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"New folder created: {new_folder_path}")
        else:
            print(f"Folder already exists: {new_folder_path}")
        speed_time = str(datetime.now().isoformat())
        idx = speed_time.index(".")
        speed_time = speed_time[idx + 1:] + ".jpg"

        url = os.path.join(new_folder_path, speed_time)

        cv2.imwrite(url, frame)
        mycol.update_one({"_id": alert_id[0]}, {"$push": {"speed_images": url}})
def day_speed(frame, mycol, alert_id):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)
    # _, binary_frame = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    # print("x:", x, "y:",y, "w:",w, "h:", h)
    # print(frame.shape[1]//2)
    tan_value = h / w
    angle_rad = math.atan(tan_value)
    angle_deg = math.degrees(angle_rad)
    # mid = frame.shape[1] // 2
    speed = 0
    if y >= 226:
        speed = 10 - angle_deg * 90/180
    elif x < 287:
        speed = angle_deg * 90/180 + 12
    else:
        speed = (90 - angle_deg) * 90 / 180 + 55

    cv2.putText(frame, f'speed: {speed}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)



    # Set Vietnam's time zone


    # Convert GMT time to Vietnam time
    result = mycol.find_one({"_id": alert_id[0]})
    if result:
        print("Document found:")
        mycol.update_one({"_id": alert_id[0]}, {"$set": {"speed": speed}})
        new_folder_path = "D:\\speed\\" + str(alert_id[0])
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"New folder created: {new_folder_path}")
        else:
            print(f"Folder already exists: {new_folder_path}")
        speed_time = str(datetime.now().isoformat())
        idx = speed_time.index(".")
        speed_time = speed_time[idx+1:] +".jpg"
        url = os.path.join(new_folder_path, speed_time)

        cv2.imwrite(url, frame)
        mycol.update_one({"_id": alert_id[0]}, {"$push": {"speed_images": url}})


def consume_video_from_kafka_1(consumer, topic):
    # bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
    flag = [0]
    for message in consumer:
        frame_data = np.frombuffer(message.value, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        yolo_dectect(model, frame, flag, mycol,alert_id)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




def consume_video_from_kafka2(consumer, topic, is_night = False):

    for message in consumer:
        frame_data = np.frombuffer(message.value, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        if is_night:
            night_speed(frame,mycol, alert_id)
        else:
            day_speed(frame, mycol, alert_id)
        cv2.imshow(topic, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    topic1 = 'topic1'
    topic2 = 'topic2'
    consumer1 = KafkaConsumer(
        topic1,
        bootstrap_servers=['localhost:9092'],
        api_version=(0, 10)
    )
    consumer2 = KafkaConsumer(
        topic2,
        bootstrap_servers=['localhost:9092'],
        api_version=(0, 10)
    )
    consumer_thread1 = threading.Thread(target=consume_video_from_kafka_1,
                                        args=(consumer1, topic1))
    consumer_thread2 = threading.Thread(target=consume_video_from_kafka2,
                                        args=(consumer2, topic2))

    consumer_thread1.start()
    consumer_thread2.start()

    consumer_thread2.join()
    consumer_thread1.join()


if __name__ == "__main__":
    main()
