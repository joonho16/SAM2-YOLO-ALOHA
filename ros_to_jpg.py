#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import numpy as np
import glob
import os
import sys

from utils import zoom_image

from constants import TASK_CONFIGS

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


def main(cam_name, output_dir, task_name):

    os.makedirs(output_dir, exist_ok=True)

    rospy.init_node('image_subscriber', anonymous=True)
    subscriber = ImageSubscriber(f'/{cam_name}/color/image_raw')

    print(TASK_CONFIGS[task_name]['camera_config'])   

    camera_config = TASK_CONFIGS[task_name]['camera_config']

    existing_images = sorted(glob.glob(os.path.join(output_dir, '*.jpg')))
    image_count = len(existing_images)  # 다음 저장 번호

    rate = rospy.Rate(10)  # 초당 30회 루프 실행

    is_recording = False

    while not rospy.is_shutdown():
        image = subscriber.image
        if image is not None:
            if cam_name in camera_config:
                if 'zoom' in camera_config[cam_name]:
                    size = camera_config[cam_name]['zoom']['size']
                    point = camera_config[cam_name]['zoom']['point']
                    image = zoom_image(image, point, size)
                if 'resize' in camera_config[cam_name]:
                    size = camera_config[cam_name]['resize']['size']
                    image = cv2.resize(image, size)

            # 이미지 크기 조정 후 띄우기
            max_height = 480
            max_width = 640
            resized_image = cv2.resize(image, (max_width, max_height))
            cv2.imshow("Image", resized_image)

            # 키 입력 체크
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # 'q' 키로 종료
                break
            elif key == ord(' '):
                # 스페이스바를 누르면 저장 on/off 토글
                is_recording = not is_recording
                rospy.loginfo(f"Recording mode: {is_recording}")

            # 저장 모드(is_recording)가 True일 때만 저장
            if is_recording:
                filename = f"{image_count:05d}.jpg"
                save_path = os.path.join(output_dir, filename)
                success = cv2.imwrite(save_path, image)
                if not success:
                    rospy.logerr(f"Failed to save image at {save_path}")
                else:
                    rospy.loginfo(f"Saved image: {save_path}")
                image_count += 1

        else:
            print("No images to display.")

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam_name = "camera1"
    output_dir = "yolo/raw_data/tomato_data"
    task_name = "pick_tomato"
    main(cam_name, output_dir, task_name)