#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import numpy as np

from utils import zoom_image, resize_image
from constants import TASK_CONFIGS

from ultralytics import YOLO
from apply_yolo import mask_outside_boxes

yolo_model = YOLO('yolo/runs/detect/train2/weights/best.pt')

class ImageSubscriber:

    def __init__(self, topic_name):
        self.bridge = CvBridge()
        self.cam_name = topic_name
        self.image = None
        self.lock = threading.Lock()
        self.image_sub = rospy.Subscriber(topic_name, Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.ros_image_to_numpy(data)
            with self.lock:
                self.image = cv_image
        except CvBridgeError as e:
            print(e)

    def ros_image_to_numpy(self, image_msg):
        # 이미지의 데이터 타입 추출
        encoding_to_dtype = {
            'rgb8': ('uint8', 3),
            'bgr8': ('uint8', 3),
            'mono8': ('uint8', 1),
            'mono16': ('uint16', 1),
            'rgba8': ('uint8', 4),
            'bgra8': ('uint8', 4),
        }

        if image_msg.encoding not in encoding_to_dtype:
            raise ValueError(f"Unsupported encoding: {image_msg.encoding}")
        
        dtype, channels = encoding_to_dtype[image_msg.encoding]
        
        # NumPy 배열 생성
        data = np.frombuffer(image_msg.data, dtype=dtype)

        image_array = data.reshape((image_msg.height, image_msg.width, channels))
        
        # RGB와 BGR 간 변환
        if image_msg.encoding in ['rgb8', 'bgr8']:
            image_array = image_array[:, :, ::-1]  # BGR -> RGB
        elif image_msg.encoding == ['rgb8', 'bgra8']:
            image_array = image_array[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
        return image_array


def main():
    rospy.init_node('image_subscriber', anonymous=True)
    task_name = 'pick_tomato'
    subscribers = []
    for cam_name in TASK_CONFIGS[task_name]['camera_names']:
        subscribers.append(ImageSubscriber(f'/{cam_name}/color/image_raw'))
    camera_config = TASK_CONFIGS[task_name]['camera_config']


    rate = rospy.Rate(30)  # 초당 30회 루프 실행
    while not rospy.is_shutdown():
        images = []
        for index, cam_name in enumerate(TASK_CONFIGS[task_name]['camera_names']):
            image = subscribers[index].image
            if image is not None:
                if cam_name in camera_config:
                    if 'masked_yolo' in camera_config[cam_name]:
                        masked_image = np.zeros_like(image)
                        if camera_config[cam_name]['masked_yolo']['is_fixed_mask']:
                            boxes = []
                            if is_first_img:     
                                results = yolo_model(image)
                                for result in results:
                                    boxes = result.boxes
                                    classes = result.names
                            if len(boxes) > 0:
                                masked_image = mask_outside_boxes(image, boxes, padding=10, show_all=False, index=0)
                                is_first_img = False
                        else:
                            results = yolo_model(image)
                            for result in results:
                                boxes = result.boxes
                                classes = result.names
                            masked_image = mask_outside_boxes(image, boxes, padding=10, show_all=False, index=9)
                        image = masked_image
                    if 'zoom' in camera_config[cam_name]:
                        zoom_factor = camera_config[cam_name]['zoom']['rate']
                        point = camera_config[cam_name]['zoom']['point']
                        image = zoom_image(image, zoom_factor, point)
                    if 'resize' in camera_config[cam_name]:
                        resize_rate = camera_config[cam_name]['resize']['rate']
                        image = resize_image(image, resize_rate)
                    images.append(image)
        if images:
            
            # 이미지 크기 맞추기 (최대 크기로 맞추거나 다른 방식으로 조정)
            max_height = 480
            max_width = 640
            resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

            # 이미지를 가로로 나열
            combined_image = cv2.hconcat(resized_images)

            # 단일 창에 표시
            cv2.imshow("Combined Image", combined_image)
        else:
            print("No images to display.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
