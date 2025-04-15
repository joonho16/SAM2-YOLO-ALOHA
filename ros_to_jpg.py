#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import numpy as np
import glob
import os
import sys

from utils import zoom_image, ros_image_to_numpy
from constants import TASK_CONFIGS

import subprocess


class ImageSubscriber:

    def __init__(self, topic_name):
        self.bridge = CvBridge()
        self.cam_name = topic_name
        self.image = None
        self.lock = threading.Lock()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.callback)

    def callback(self, data):
        try:
            cv_image = ros_image_to_numpy(data)
            with self.lock:
                self.image = cv_image
        except CvBridgeError as e:
            print(e)


def main():
    # 저장할 카메라 이름들
    camera_list = ["camera1", "camera2"]
    
    # 예시 task_name
    task_name = "pick_tomato"
    
    # 저장할 폴더(베이스)와 카메라별 세부 폴더
    base_output_dir = "tomato_video"
    
    # 카메라별 ImageSubscriber 객체 생성
    subscribers = {}
    for cam_name in camera_list:
        topic = f"/{cam_name}/color/image_raw/compressed"
        rospy.loginfo(f"Subscribe topic: {topic}")
        subscribers[cam_name] = ImageSubscriber(topic)

    # 각 카메라별로 output_dir 준비 + 이미지 카운트 체크
    image_counts = {}
    for cam_name in camera_list:
        output_dir = os.path.join(base_output_dir, cam_name)
        os.makedirs(output_dir, exist_ok=True)
        existing_images = sorted(glob.glob(os.path.join(output_dir, '*.jpg')))
        image_count = len(existing_images)  # 다음 저장 번호
        image_counts[cam_name] = {"output_dir": output_dir, "count": image_count}
    
    rospy.init_node('image_subscriber_multi', anonymous=True)

    # 카메라 config (예시)
    camera_config = TASK_CONFIGS[task_name]['camera_config']  # dict 형태라고 가정

    rate = rospy.Rate(20)  # 초당 10회 루프
    is_recording = False

    while not rospy.is_shutdown():
        # 각각의 카메라 이미지를 순회하며 화면 표시 + 저장
        for cam_name in camera_list:
            subscriber = subscribers[cam_name]
            with subscriber.lock:
                image = subscriber.image

            if image is not None:
                # 필요하면 config에 따라 zoom/resize 적용
                if cam_name in camera_config:
                    if 'zoom' in camera_config[cam_name]:
                        size = camera_config[cam_name]['zoom']['size']
                        point = camera_config[cam_name]['zoom']['point']
                        image = zoom_image(image, point, size)
                    if 'resize' in camera_config[cam_name]:
                        size = camera_config[cam_name]['resize']['size']
                        image = cv2.resize(image, size)

                # 화면에 표시 (크기가 너무 크면 리사이즈 후 표시)
                max_height = 480
                max_width = 640
                disp_image = cv2.resize(image, (max_width, max_height))
                cv2.imshow(f"Image {cam_name}", disp_image)

                # 저장 모드(is_recording)가 True라면 저장
                if is_recording:
                    output_dir = image_counts[cam_name]["output_dir"]
                    count = image_counts[cam_name]["count"]
                    filename = f"{count:05d}.jpg"
                    save_path = os.path.join(output_dir, filename)
                    success = cv2.imwrite(save_path, image)
                    if not success:
                        rospy.logerr(f"[{cam_name}] Failed to save image at {save_path}")
                    else:
                        rospy.loginfo(f"[{cam_name}] Saved image: {save_path}")
                    image_counts[cam_name]["count"] += 1

        # 키 입력 체크 (OpenCV 윈도우 중 하나가 포커스를 가져야 키 입력 동작)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 'q' 키로 종료
            break
        elif key == ord(' '):
            # 스페이스바를 누르면 저장 on/off 토글
            is_recording = not is_recording
            rospy.loginfo(f"Recording mode: {is_recording}")

        rate.sleep()

    cv2.destroyAllWindows()

    for cam_name in camera_list:
        image_folder = f"{base_output_dir}/{cam_name}"
        output_video = f"{base_output_dir}/{cam_name}.mp4"
        cmd = [
            "ffmpeg", "-framerate", "20",
            "-i", os.path.join(image_folder, "%05d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
        ]

        subprocess.run(cmd)
        print("MP4 변환 완료!")

if __name__ == '__main__':
    main()