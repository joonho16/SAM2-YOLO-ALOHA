import cv2
import os
import hydra
import h5py
import shutil
import random


import numpy as np
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
import sam2
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor
from constants import YOLO_CONFIG

from numba import cuda
import torch
import yaml
import yolo.data_augmentation as yolo_dag
import json

import readline
from ultralytics import YOLO


class YoloViaSam2:

    def __init__(self, task_name, config):
        self.class_names = config['class_names']
        self.checkpoint = config['checkpoint']
        self.sam2_config_dir = config['sam2_config_dir']
        self.model_cfg_yaml = config['model_cfg_yaml']
        self.raw_data_dir = config['raw_data_dir']
        self.data_dir = config['data_dir']
        self.tmp_dir = config['tmp_data_dir']
        self.yaml_dir = config['yaml_dir']
        self.task_name = task_name
        self.image_view_size = 800
        self.resize_rate = 1

        self.point_groups = []
        self.label_groups = []
        self.class_ids = []

    def video_to_jpg(self, video_path, save_dir, video_type, scale):
        if video_type == 'mp4':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 동영상 불러오기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("동영상을 열 수 없습니다.")
                return

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # 더 이상 프레임이 없으면 종료

                # 해상도 낮추기
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))

                # 파일 이름 정하기
                file_name = os.path.join(save_dir, f"{frame_count:05d}.jpg")

                # 이미지 저장 (JPG 형식)
                cv2.imwrite(file_name, frame_resized)

                frame_count += 1

            cap.release()
            print(f"총 {frame_count}개의 프레임이 {save_dir}에 저장되었습니다.")


    def hdf5_to_jpg(self, hdf5_path, save_dir, cam_name, start_point=0):
        os.makedirs(save_dir, exist_ok=True)
        with h5py.File(hdf5_path, 'r') as f:
            images = f[f"observations/images/{cam_name}"]

            episode_len = len(images)
            num = 0
            for i in range(start_point, episode_len):
                file_name = os.path.join(save_dir, f"{num:05d}.jpg")

                success = cv2.imwrite(file_name, images[i])
                if not success:
                    print(f"Failed to save image at {save_dir}")
                else:
                    print(f"Saved image: {save_dir}")
                num += 1

            print(f"총 {num}개의 프레임이 {save_dir}에 저장되었습니다.")


    def set_sam2_prompt(self, image, class_num):
        point_group = []
        point_groups = [[]]
        label_group = []
        label_groups = [[]]
        class_id = 0
        class_ids = [0]
        group_index = 0

        def log():
            print("Point Groups: ", point_groups)
            print("Label Groups: ", label_groups)
            print("Class_ids: ", class_ids)
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
                label_groups[group_index].append(1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                label_groups[group_index].append(0)

            if event == event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                point_groups[group_index].append([x // self.resize_rate, y // self.resize_rate])
                log()

        if img is None:
            print("Error: Unable to load image.")
        else:
            # 창 이름 설정
            window_name = 'Image Viewer'
            if image.shape[0] < self.image_view_size:
                self.resize_rate = self.image_view_size // image.shape[0]

            resized_image = cv2.resize(image, (image.shape[1] * self.resize_rate, image.shape[0] * self.resize_rate))
            
            cv2.imshow(window_name, resized_image)

            # 클릭 이벤트 연결
            cv2.setMouseCallback(window_name, click_event)

            # ESC 키를 누를 때까지 대기
            print("Press ESC to exit.")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter 키
                    break
                if key == 32:  # Space 키
                    point_group = []
                    point_groups.append(point_group)
                    label_group = []
                    label_groups.append(label_group)
                    group_index += 1
                    class_ids.append(class_id)
                    log()
                if key == ord('c') or key == ord('C'):
                    if class_ids[group_index] == class_num - 1:
                        class_ids[group_index] = 0
                    else:
                        class_ids[group_index] += 1
                    log()
                if key == 27:  # ESC 키
                    point_group = []
                    point_groups = [[]]
                    label_group = []
                    label_groups = [[]]
                    class_id = 0
                    class_ids = [0]
                    group_index = 0
                    log()

            # 클릭된 좌표 출력
            # 창 닫기
            cv2.destroyAllWindows()

        return point_groups, label_groups, class_ids
    

    def save_yolo_labels(self, bboxes, image_width, image_height, output_file, class_id=0):
        with open(output_file, 'w') as file:
            for bbox in bboxes:
                x_min, y_min, x_max, y_max, class_id = bbox

                # Calculate YOLO format values
                x_center = ((x_min + x_max) / 2) / image_width
                y_center = ((y_min + y_max) / 2) / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Write to file
                file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


    def calculate_bounding_box(self, outmask):
        """
        Calculate the bounding box of the region where outmask is 1.

        Parameters:
            outmask (np.ndarray): Input mask with shape (1, 320, 320).

        Returns:
            tuple: (x_min, y_min, x_max, y_max) representing the bounding box.
        """
        if outmask.shape[0] != 1:
            raise ValueError("Input outmask must have shape (1, 320, 320)")

        # Remove the first dimension
        mask = outmask[0]

        # Find indices where the mask is 1
        indices = np.argwhere(mask == 1)

        if indices.size == 0:
            # If no region is found, return None
            return None

        # Get minimum and maximum coordinates
        y_min, x_min = indices.min(axis=0)
        y_max, x_max = indices.max(axis=0)

        return x_min, y_min, x_max, y_max


    def sam2_to_bounding_box(self, jpg_dir, label_dir, point_groups, label_groups, class_ids):
            # Hydra 설정 경로 초기화
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # 기존 설정 초기화
        hydra.initialize(config_path=self.sam2_config_dir, version_base=None)

        # 모델 생성 및 예측
        predictor = build_sam2_video_predictor(self.model_cfg_yaml, self.checkpoint)

        np.random.seed(3)

        frame_names = [
            p for p in os.listdir(jpg_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=jpg_dir)

        predictor.reset_state(inference_state)

        ann_frame_idx = 0

        for (index, point) in enumerate(point_groups):
            # Let's add a positive click at (x, y) = (210, 350) to get started
            point_group = np.array([point], dtype=np.float32)
            label_group = np.array([label_groups[index]], np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=index,
                points=point_group,
                labels=label_group,
            )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        # 비디오로 세그먼트
        vis_frame_stride = len(frame_names) // 5
        # Main code
        for out_frame_idx in range(0, len(frame_names)):
            image_path = os.path.join(jpg_dir, frame_names[out_frame_idx])
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {image_path}")
                continue

            height, width, _ = image.shape

            boxes = []
            combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                class_id = class_ids[out_obj_id] 
                box = self.calculate_bounding_box(out_mask)
                if box is None:
                    continue
                else:
                    x_min, y_min, x_max, y_max = box
                boxes.append([x_min, y_min, x_max, y_max, class_id])

                if out_frame_idx % vis_frame_stride == 0:
                    mask = self.get_mask(out_mask, class_id)
                    combined_mask = np.maximum(combined_mask, mask)

            masked_image = cv2.addWeighted(combined_mask, 0.5, image, 0.5, 0)
            if out_frame_idx % vis_frame_stride == 0:
                cv2.imshow("Combined Image", cv2.resize(masked_image, (masked_image.shape[1] * self.resize_rate, masked_image.shape[0] * self.resize_rate)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            os.makedirs(label_dir, exist_ok=True)  # Ensure output directory exists
            output_file = os.path.join(label_dir, f"{out_frame_idx:05}.txt")
            self.save_yolo_labels(boxes, width, height, output_file)

        cv2.destroyAllWindows()

    def get_mask(self, mask, class_id):
        color_pallete = [
            [255, 0, 0],   # 빨강
            [0, 255, 0],   # 초록
            [0, 0, 255],   # 파랑
            [255, 255, 0], # 노랑
            [255, 0, 255], # 자홍
            [0, 255, 255], # 청록
            [128, 0, 0],   # 어두운 빨강
            [0, 128, 0],   # 어두운 초록
            [0, 0, 128],   # 어두운 파랑
            [128, 128, 0]  # 올리브색
        ]
        colored_image = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

        # True 부분에 색을 입히기
        colored_image[mask[0]] = color_pallete[class_id]

        return colored_image
    
    def merge_folders(self, src_folder, src_labeled_folder):
        
        os.makedirs(self.tmp_dir + '/' + self.task_name, exist_ok=True)

        tmp_folder = self.tmp_dir + '/' + self.task_name + '/image'
        tmp_label_folder = self.tmp_dir + '/' + task_name + '/label'

                # 병합 폴더 생성
        os.makedirs(tmp_folder, exist_ok=True)
        os.makedirs(tmp_label_folder, exist_ok=True)

        count = len(os.listdir(tmp_folder))

        for filename in sorted(os.listdir(src_folder)):
            if filename.endswith(".jpg"):
                # 이미지 파일 복사
                src_image_path = os.path.join(src_folder, filename)
                dst_image_path = os.path.join(tmp_folder, f"{count:05}.jpg")
                shutil.copy(src_image_path, dst_image_path)
                
                # 라벨 파일 복사
                label_filename = filename.replace(".jpg", ".txt")
                src_label_path = os.path.join(src_labeled_folder, label_filename)
                dst_label_path = os.path.join(tmp_label_folder, f"{count:05}.txt")
                if os.path.exists(src_label_path):
                    shutil.copy(src_label_path, dst_label_path)
                
                count += 1

    def data_augment(self):
        yolo_dag.main(self.tmp_dir + '/' + self.task_name + '/image', self.tmp_dir + '/' + self.task_name + '/label')

    def data_align(self):
        """
        데이터 폴더(data_name)와 라벨 폴더(data_name_labeled)에서
        파일 이름은 다르지만, 순서가 1:1 대응되어 있다고 가정할 때,
        train/valid/test 비율로 나누어 복사하는 함수.
        """

        # 1. 출력할 폴더 생성
        base_output_dir = os.path.join(self.data_dir, self.task_name)

        # train, valid, test 각각의 images, labels 폴더
        train_images_output_dir = os.path.join(base_output_dir, "train", "images")
        train_labels_output_dir = os.path.join(base_output_dir, "train", "labels")
        valid_images_output_dir = os.path.join(base_output_dir, "valid", "images")
        valid_labels_output_dir = os.path.join(base_output_dir, "valid", "labels")
        test_images_output_dir  = os.path.join(base_output_dir, "test",  "images")
        test_labels_output_dir  = os.path.join(base_output_dir, "test",  "labels")
        
        os.makedirs(train_images_output_dir, exist_ok=True)
        os.makedirs(train_labels_output_dir, exist_ok=True)
        os.makedirs(valid_images_output_dir, exist_ok=True)
        os.makedirs(valid_labels_output_dir, exist_ok=True)
        os.makedirs(test_images_output_dir,  exist_ok=True)
        os.makedirs(test_labels_output_dir,  exist_ok=True)

        # 2. 원본 데이터 경로 및 라벨 경로 지정
        image_dir = self.tmp_dir + '/' + self.task_name + '/image'
        label_dir = self.tmp_dir + '/' + self.task_name + '/label'

        # 3. 파일 목록 불러오기 (정렬 후 리스트화)
        data_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

        # 4. 데이터 개수 확인 후, 둘 중 더 적은 쪽까지 페어링
        total_pairs = min(len(data_files), len(label_files))

        # 5. 인덱스 리스트 만들어 랜덤 셔플
        indices = list(range(total_pairs))
        random.shuffle(indices)

        # 6. 전체 파일 개수에 따라 train / valid / test 분할
        train_data_len = total_pairs * 2 // 3
        valid_data_len = (total_pairs - train_data_len) // 2
        test_data_len = total_pairs - train_data_len - valid_data_len

        train_indices = indices[:train_data_len]
        valid_indices = indices[train_data_len:train_data_len + valid_data_len]
        test_indices  = indices[train_data_len + valid_data_len:]

        # 7. 복사 함수 정의
        def copy_files(index_list, images_dir, labels_dir):
            for idx in index_list:
                # 매칭되는 data_file, label_file
                data_file = data_files[idx]
                label_file = label_files[idx]

                # 원본 경로
                src_data_path  = os.path.join(image_dir, data_file)
                src_label_path = os.path.join(label_dir, label_file)

                # 복사될 경로 (이미지/라벨 각각 분리)
                dst_data_path  = os.path.join(images_dir, data_file)
                dst_label_path = os.path.join(labels_dir, label_file)

                shutil.copy2(src_data_path,  dst_data_path)
                shutil.copy2(src_label_path, dst_label_path)

        # 8. 데이터 분할별 복사 (images/labels 각각)
        copy_files(train_indices, train_images_output_dir, train_labels_output_dir)
        copy_files(valid_indices, valid_images_output_dir, valid_labels_output_dir)
        copy_files(test_indices,  test_images_output_dir,  test_labels_output_dir)

        # 9. 개수 출력
        print(f"Completed alignment for {self.task_name}:")
        print(f" - Train: {len(train_indices)} pairs")
        print(f" - Valid: {len(valid_indices)} pairs")
        print(f" - Test:  {len(test_indices)} pairs")

    
    def make_yaml(self):
        # root = f'{self.data_dir}/{self.task_name}'.replace(self.yaml_dir, '')
        root = 'data/' + self.task_name
        data = {
            "train" : f'{root}/train/',
            "val" : f'{root}/valid/',
            "test" : f'{root}/test/',
            "names" : { i : class_name for i, class_name in enumerate(self.class_names) }
        }

        with open(f'{self.yaml_dir}/{self.task_name}.yaml', 'w') as f :
            yaml.dump(data, f)

        # check written file
        with open(f'{self.yaml_dir}/{self.task_name}.yaml', 'r') as f :
            lines = yaml.safe_load(f)
            print(lines)

def input_caching(prompt):
    cache_file_path = "input_cache.json"
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    default = cache.get(prompt, "")
    def prefill_hook():
        readline.insert_text(default)  # 기본값 입력
        readline.redisplay()          # 화면에 표시
    readline.set_pre_input_hook(prefill_hook)

    answer = input(prompt)

    cache[prompt] = answer

    with open(cache_file_path, "w") as f:
        json.dump(cache, f, indent=4)

    return answer


if __name__ == '__main__':

    task_name = input_caching('Enter Task Name: ')
    config = YOLO_CONFIG[task_name]
    yolo_via_sam2 = YoloViaSam2(task_name, config)

    folder_names = input_caching('Enter JPG Data Folder Name (Space for Multi-Input): ')

    start_num = 0
    for index, folder_name in enumerate(folder_names.split(" ")):
        jpg_dir = config['raw_data_dir'] + '/' + folder_name
        label_dir = jpg_dir + '_label'
        print(f'------For Folder { folder_name }------')
        data_type = input_caching(f'Enter Image Data Type for {folder_name} (mp4, rostopic, hdf5): ')

        if data_type == 'mp4':
            file_path = input_caching(f'Enter Video File Path for {folder_name} (Press Enter to Skip it.): ')

            if file_path != '':
                yolo_via_sam2.video_to_jpg(file_path, jpg_dir, 'mp4', 0.5)

        elif data_type == 'rostopic':
            pass

        elif data_type == 'hdf5':
            file_path = input_caching(f'Enter Hdf5 File Path for {folder_name} (Press Enter to Skip it.): ')
            if file_path != '':
                cam_name = input_caching(f'Enter Camera Name for {folder_name} (Ex. camera1 / camera2): ')
                start_point = int(input_caching(f'Enter Start Point for {folder_name}: '))
                yolo_via_sam2.hdf5_to_jpg(file_path, jpg_dir, cam_name, start_point)

        answer = input(f'Sam2 to Bounding Box for {folder_name} (Press "y" to process it.): ')

        if answer == 'y':
            image_path = jpg_dir + '/00000.jpg'
            img = cv2.imread(image_path)
            point_groups, label_groups, class_ids = yolo_via_sam2.set_sam2_prompt(img, 2)

            yolo_via_sam2.sam2_to_bounding_box(jpg_dir, label_dir, point_groups, label_groups, class_ids)

        answer = input_caching(f'Data to tmp for {folder_name} (Press "y" to process it.): ')

        if answer == 'y':
            yolo_via_sam2.merge_folders(jpg_dir, label_dir)

        torch.cuda.empty_cache()

    # answer = input('Data Augmentation: (Press "y" to process it.): ')

    # if answer == 'y':
    #     yolo_via_sam2.data_augment()

    answer = input_caching('Dataset Align: (Press "y" to process it.): ')
    
    if answer == 'y':
        yolo_via_sam2.data_align()

    answer = input_caching('Make Yaml File: (Press "y" to process it.): ')
    
    if answer == 'y':
        yolo_via_sam2.make_yaml()

    answer = input_caching('Train YOLO: (Press "y" to process it.): ')
    
    if answer == 'y':
        model = YOLO(config['yolo_path'])
        model.train(data=f'{yolo_via_sam2.yaml_dir}/{task_name}.yaml' , epochs=config['epochs'])
    