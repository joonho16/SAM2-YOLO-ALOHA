#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import numpy as np

from utils import zoom_image
from constants import TASK_CONFIGS

from ultralytics import YOLO
from apply_yolo import mask_outside_boxes


class ImageSubscriber:

    def __init__(self, topic_name):
        self.bridge = CvBridge()
        self.cam_name = topic_name
        self.image = None
        self.lock = threading.Lock()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.callback)

    def callback(self, data):
        try:
            cv_image = self.ros_image_to_numpy(data)
            with self.lock:
                self.image = cv_image
        except CvBridgeError as e:
            print(e)

    def ros_image_to_numpy(self, image_msg):
        if isinstance(image_msg, CompressedImage):
            # 압축 이미지 처리
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 기본 BGR 형태로 디코딩됨
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)  # RGB로 변환
            image_array = image_array[:, :, ::-1]  # BGR -> RGB
            return image_array

        # 일반 Image 메시지 처리
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
        data = np.frombuffer(image_msg.data, dtype=dtype)
        image_array = data.reshape((image_msg.height, image_msg.width, channels))
        
        if image_msg.encoding == 'bgr8':
            image_array = image_array[:, :, ::-1]  # BGR -> RGB
        elif image_msg.encoding == 'bgra8':
            image_array = image_array[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
        
        return image_array

def main():
    rospy.init_node('image_subscriber', anonymous=True)
    task_name = 'grasp_cable_yaskawa'
    subscribers = []
    for cam_name in TASK_CONFIGS[task_name]['camera_names']:
        subscribers.append(ImageSubscriber(f'/{cam_name}/color/image_raw/compressed'))
    camera_config = TASK_CONFIGS[task_name]['camera_config']


    rate = rospy.Rate(30)  # 초당 30회 루프 실행
    while not rospy.is_shutdown():
        images = []
        for index, cam_name in enumerate(TASK_CONFIGS[task_name]['camera_names']):
            image = subscribers[index].image
            if image is not None:
                if cam_name in camera_config:
                    if 'zoom' in camera_config[cam_name]:
                        size = camera_config[cam_name]['zoom']['size']
                        point = camera_config[cam_name]['zoom']['point']
                        image = zoom_image(image, point, size)
                    if 'resize' in camera_config[cam_name]:
                        size = camera_config[cam_name]['resize']['size']
                        image = cv2.resize(image, size)
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
