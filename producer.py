import cv2
from kafka import  KafkaProducer
import threading

def publish_video_to_kafka(producer, topic, video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        producer.send(topic, value = data)
    cap.release()

def main():

    bootstrap_servers = 'localhost:9092'
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        api_version=(0,10,1)
    )

    topic1 = 'topic1'
    video_path1 = "check.mp4"

    producer_thread1 = threading.Thread(target=publish_video_to_kafka,
                                        args=(producer, topic1, video_path1))
    producer_thread1.start()
    topic2 = 'topic2'
    video_path2 = 'speed.mp4'

    producer_thread2 = threading.Thread(target=publish_video_to_kafka,
                                        args=(producer, topic2, video_path2))
    producer_thread2.start()

    producer_thread2.join()
    producer_thread1.join()

if __name__ == "__main__" :
    print("main ne")
    main()



