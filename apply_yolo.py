from ultralytics import YOLO
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import numpy as np
import h5py
import sys
import os
import time

from utils import zoom_image, fetch_image_with_config
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

def mask_outside_boxes(image, boxes_list, padding=0):
    """
    박스 내부 이미지만 살리고, 나머지 영역은 검게 칠하는 함수.
    """

    height, width, _ = image.shape
    # 원본 이미지와 동일한 크기의 흰색 이미지 초기화
    masked_image = np.full_like(image, 255)

    for boxes in boxes_list:

        # YOLO 박스에서 xyxy 좌표 가져오기 (tensor 형태)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) 형태의 NumPy 배열로 변환

        # 박스별로 반복하며 박스 영역을 복사
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            masked_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return masked_image


def hdf5_to_yolo_image(model, width, height, task_name):

    # 기본값 설정

    episode_len = TASK_CONFIGS[task_name]['episode_len']
    num_episodes = TASK_CONFIGS[task_name]['num_episodes']
    # # HDF5 파일 경로
    # hdf5_path = f"{dir}/{work}/original/episode_{episode}.hdf5"

    for episode in range(0, num_episodes):

        hdf5_path = f"{TASK_CONFIGS[task_name]['dataset_dir']}/original/episode_{episode}.hdf5"
    
        # # hdf5 사용하기      
        with h5py.File(hdf5_path, 'r') as f:
            images = []
            
            data_dict = {
                '/observations/qpos': f[f'/observations/qpos'],
                '/observations/qvel': f[f'/observations/qvel'],
                '/observations/effort': f[f'/observations/effort'],
                '/action': f[f'action'],
            }

            for im_name in TASK_CONFIGS[task_name]['camera_names']:

                camera_config = TASK_CONFIGS[task_name]['camera_config'][im_name]
                image = f[f"observations/images/{im_name}"]

                if 'masked_yolo' not in camera_config:
                    data_dict[f'/observations/images/{im_name}'] = image
                    continue


                frames = np.transpose(image, (0, 1, 2, 3))

                classes = camera_config['masked_yolo']['classes']

                is_first_img = True

                fixed_boxes = None

                for frame in frames:

                    all_boxes = []

                    for class_name, config in classes.items():

                        masked_image = np.zeros_like(frame)
                        resized_image = cv2.resize(frame, (width, height))
                        
                        if config['is_fixed_mask']:

                            if is_first_img:
                                results = model(resized_image)
                                result = results[0]
                                fixed_boxes = result.boxes
                                names = result.names

                                # print(result.names[config['id']])

                                if class_name == result.names[config['id']]:

                                    # 같은 물체가 여러 개일 때, 내가 보여주고 싶은 물체 id 특정
                                    if config['show_id'] != -1 and len(fixed_boxes) > 0: # -1: 모두 선택
                                        fixed_boxes = fixed_boxes[config['show_id']]

                                    is_first_img = False

                            if fixed_boxes != None and len(fixed_boxes) > 0:
                                all_boxes.append(fixed_boxes)

                        else:
                            print(config['is_fixed_mask'])
                            results = model(resized_image, conf=0.2)
                            result = results[0]
                            boxes = result.boxes
                            names = result.names

                            for box in boxes:

                                box_id = int(box.cls.item())

                                if len(boxes) > 0:

                                    if box_id == config['id']:
                                        # 같은 물체가 여러 개일 때, 내가 보여주고 싶은 물체 id 특정
                                        if config['show_id'] != -1: # -1: 모두 선택
                                            boxes = boxes[config['show_id']]

                                        all_boxes.append(boxes)

                    if len(all_boxes) > 0:
                        masked_image = mask_outside_boxes(resized_image, all_boxes, padding=5)
                            # if len(boxes) > 0:
                            #     masked_image = mask_outside_boxes(resized_image, boxes, class_name, padding=15, show_all=False, index=0)

                            #     if is_first_img:
                            #         print(f"Episode: {episode}")
                            #         cv2.imshow('Masked Image', masked_image)
                            #         cv2.waitKey(2000)
                            #         cv2.destroyAllWindows()

                        # 마스킹 된 화면 보기
                        cv2.imshow('Masked Image', masked_image)
                        cv2.waitKey(100)


                    masked_image = cv2.resize(masked_image, (160, 120))

                    data_dict[f'/observations/images/{im_name}'] = masked_image

                cv2.destroyAllWindows()

            dataset_path = f"{TASK_CONFIGS[task_name]['dataset_dir']}/yolo/episode_{episode}.hdf5"
            with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in ['camera1', 'camera2']:
                    _ = image.create_dataset(cam_name, (episode_len, 120, 160, 3), dtype='uint8',
                                            chunks=(1, 120, 160, 3), )

                _ = obs.create_dataset('qpos', (episode_len, 7))
                _ = obs.create_dataset('qvel', (episode_len, 7))
                _ = obs.create_dataset('effort', (episode_len, 7))
                _ = root.create_dataset('action', (episode_len, 7))
            
                for name, array in data_dict.items():
                    root[name][...] = array

                # # 마스킹 없이 보기
                # annotated_frame = result.plot()
                # cv2.imshow("Combined Image", annotated_frame)

                # 마스킹 된 화면 보기
                cv2.imshow('Masked Image', masked_image)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()



def camera_to_yolo_image(task_name, model, width, height, is_fixed_mask=False):
    rospy.init_node('image_subscriber', anonymous=True)
    subscribers = []
    for cam_name in TASK_CONFIGS[task_name]['camera_names']:
        subscribers.append(ImageSubscriber(f'/{cam_name}/color/image_raw'))
    camera_config = TASK_CONFIGS[task_name]['camera_config']


    rate = rospy.Rate(30)  # 초당 30회 루프 실행
    is_first_img = True
    while not rospy.is_shutdown():
        images = []
        for index, cam_name in enumerate(TASK_CONFIGS[task_name]['camera_names']):
            image = subscribers[index].image
            if image is not None:
                if cam_name in camera_config:
                    if 'zoom' in camera_config[cam_name]:
                        zoom_factor = camera_config[cam_name]['zoom']['ratio']
                        point = camera_config[cam_name]['zoom']['point']
                        image = zoom_image(image, zoom_factor, point)
                    if 'resize' in camera_config[cam_name]:
                        resize_size = camera_config[cam_name]['resize']['size']
                        image = cv2.resize(image, resize_size)
                    images.append(image)
        if images:
            
            # 이미지 크기 맞추기 (최대 크기로 맞추거나 다른 방식으로 조정)
            masked_image = np.zeros((height, width))
            resized_images = [cv2.resize(img, (width, height)) for img in images]

            print(resized_images[0].shape)

            if is_fixed_mask:
                if is_first_img:     
                    results = model(resized_images[0])

                    for result in results:
                        boxes = result.boxes
                        classes = result.names

                if len(boxes) > 0:
                    masked_image = mask_outside_boxes(resized_images[0], boxes, padding=10)
                    is_first_img = False
                    

            else:
                results = model(resized_images[0], conf=0.25)
                for result in results:
                    boxes = result.boxes
                    classes = result.names

                masked_image = mask_outside_boxes(resized_images[0], boxes)

            # 마스킹 없이 보기
            annotated_frame = result.plot()
            cv2.imshow("Combined Image", annotated_frame)

            # # 마스킹 된 화면 보기
            # cv2.imshow('Masked Image', masked_image)

        else:
            print("No images to display.")

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        rate.sleep()

    cv2.destroyAllWindows()


def hdf5_to_yolo_image2(task_name, yolo_config):

    # 기본값 설정

    episode_len = TASK_CONFIGS[task_name]['episode_len']
    num_episodes = TASK_CONFIGS[task_name]['num_episodes']
    # # HDF5 파일 경로
    # hdf5_path = f"{dir}/{work}/original/episode_{episode}.hdf5"

    for episode in range(0, num_episodes):

        hdf5_path = f"{TASK_CONFIGS[task_name]['dataset_dir']}/original/episode_{episode}.hdf5"
    
        # # hdf5 사용하기      
        with h5py.File(hdf5_path, 'r') as f:
            masked_images = []
            
            data_dict = {
                '/observations/qpos': f[f'/observations/qpos'],
                '/observations/qvel': f[f'/observations/qvel'],
                '/observations/effort': f[f'/observations/effort'],
                '/action': f[f'action'],
            }

            for im_name in TASK_CONFIGS[task_name]['camera_names']:

                memory = None

                camera_config = TASK_CONFIGS[task_name]['camera_config'][im_name]
                images = f[f"observations/images/{im_name}"]

                if 'masked_yolo' not in camera_config:
                    data_dict[f'/observations/images/{im_name}'] = images
                    continue

                for image in images:

                    fetched_image, memory = fetch_image_with_config(image, camera_config, memory, yolo_config)

                    # 마스킹 된 화면 보기
                    cv2.imshow('Masked Image', cv2.resize(fetched_image, (640, 480)))
                    cv2.waitKey(10)

                    masked_images.append(fetched_image)
                
                data_dict[f'/observations/images/{im_name}'] = masked_images

                cv2.destroyAllWindows()

            dataset_path = f"{TASK_CONFIGS[task_name]['dataset_dir']}/yolo/episode_{episode}.hdf5"
            with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in ['camera1', 'camera2']:
                    _ = image.create_dataset(cam_name, (episode_len, 120, 160, 3), dtype='uint8',
                                            chunks=(1, 120, 160, 3), )

                _ = obs.create_dataset('qpos', (episode_len, 7))
                _ = obs.create_dataset('qvel', (episode_len, 7))
                _ = obs.create_dataset('effort', (episode_len, 7))
                _ = root.create_dataset('action', (episode_len, 7))
            
                for name, array in data_dict.items():
                    root[name][...] = array


def camera_to_fetched_image(task_name, yolo_config):
    rospy.init_node('image_subscriber', anonymous=True)
    subscribers = []
    for cam_name in TASK_CONFIGS[task_name]['camera_names']:
        subscribers.append(ImageSubscriber(f'/{cam_name}/color/image_raw'))
    camera_config = TASK_CONFIGS[task_name]['camera_config']


    rate = rospy.Rate(30)  # 초당 30회 루프 실행
    memory = None

    while not rospy.is_shutdown():
        images = []
        for index, cam_name in enumerate(TASK_CONFIGS[task_name]['camera_names']):
            image = subscribers[index].image
            if image is not None:
                if cam_name in camera_config:
                    fetched_image, memory = fetch_image_with_config(image, camera_config[cam_name], memory, yolo_config)
                    resized_image = cv2.resize(fetched_image, (640, 480))
                    images.append(resized_image)
            else:
                print("No images to display.")

            # # 마스킹 된 화면 보기
            # cv2.imshow('Masked Image', masked_image)

        if len(images) > 0:
            cv2.imshow('Images', cv2.hconcat(images))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':

    task_name = 'pick_tomato'

    yolo_config = {
        'model': YOLO('runs/detect/train17/weights/best.pt'),
        'conf': 0.4
    }

    width = 160
    height = 120
    is_fixed_mask = False
    # camera_to_yolo_image2(task_name, model, width, height, is_fixed_mask)
    # hdf5_to_yolo_image(yolo_config['model'], width, height, task_name)
    # camera_to_fetched_image(task_name, yolo_config)
    hdf5_to_yolo_image2(task_name, yolo_config)