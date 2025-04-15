from yolo_sam2 import YoloSam2
from constants import YOLO_CONFIG, TASK_CONFIGS
from utils import input_caching
import cv2
import os
import h5py
import numpy as np
import shutil
from ultralytics import YOLO

if __name__ == '__main__':
    task_name = input_caching('Enter Task Name: ')
    yolo_config = YOLO_CONFIG[task_name]
    task_config = TASK_CONFIGS[task_name]
    yolo_model = YOLO('runs/detect/train12/weights/best.pt')
    
    yolo_sam2 = YoloSam2(task_name, yolo_config)
    
    hdf5_path = input_caching(f'Enter HDF5 File Path: ')
    jpg_dir = 'tmp_for_bg_aug'
            
    background_image_path = 'bg_image/00000.jpg'
    save_dir = task_config['dataset_dir'] + '/aug/bg'
    os.makedirs(save_dir, exist_ok=True)  # Ensure output directory exists
    new_hdf5_path = save_dir + '/episode_0.hdf5'
    

    with h5py.File(hdf5_path, 'r') as f:
        with h5py.File(new_hdf5_path, 'w') as new_f:
            # 전체 그룹 구조 복사
            for key in f.keys():
                f.copy(key, new_f, name=key)

            for im_name in task_config['camera_names']:
                # background_image = cv2.imread(background_image_path)
                background_image = np.ones((120, 160, 3), dtype=np.uint8) * 255
                
                jpg_cam_dir = jpg_dir + '/' + im_name
                os.makedirs(jpg_cam_dir, exist_ok=True)  # Ensure output directory exists
                
                dataset_path = f"observations/images/{im_name}"
                images = np.array(f[dataset_path])
                vis_frame_stride = len(images) // 5
                
                cls_len = 0
                cur_index = 0
                sep_num = 0
                for index, image in enumerate(images):
                    result = yolo_model(image, conf=0.7)
                    cls = result[0].boxes.cls.cpu().numpy()
                    if len(cls) > cls_len and len(cls) < 4:
                        cls_len = len(cls)
                        if index != 0:
                            yolo_sam2.hdf5_to_jpg(hdf5_path, jpg_cam_dir + '/' + str(sep_num), im_name, cur_index, index)
                            cur_index = index
                            sep_num += 1
                            
                yolo_sam2.hdf5_to_jpg(hdf5_path, jpg_cam_dir + '/' + str(sep_num), im_name, cur_index, len(images))
                
                new_images = []
                
                segments_list = []
                for i in range(sep_num + 1):
                    jpg_cam_sep_dir = jpg_cam_dir + '/' + str(i)
                    image_path = jpg_cam_sep_dir + '/00000.jpg'
                    # img = cv2.imread(image_path)
                    
                    point_groups, label_groups, class_ids = yolo_sam2.set_sam2_prompt(img, 2)
                    
                    for point_group in point_groups:
                        for point in point_group:
                            cv2.circle(img, point, radius=3, color=(255, 0, 0), thickness=-1)
                            
                    img = cv2.resize(img, (640, 480))
                    # cv2.imshow('Image with point', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    video_segments = yolo_sam2.get_segmenatation(jpg_cam_sep_dir, point_groups, label_groups)
                    segments_list += video_segments.values()
                    
                for index, image in enumerate(images):
                    height, width, _ = image.shape
                #     combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    mask = np.zeros((1, height, width), dtype=bool)
                    for out_obj_id, out_mask in segments_list[index].items():
                        print(out_obj_id)
                        mask = out_mask | mask
                    
                    mask = mask.squeeze(axis=0)
                    mask = mask.astype(np.uint8) * 255
                    mask_3ch = cv2.merge([mask, mask, mask])
                        
                    object_part = cv2.bitwise_and(image, mask_3ch)
                    background_part = cv2.bitwise_and(background_image, cv2.bitwise_not(mask_3ch))
                    
                    result = cv2.add(object_part, background_part)
                    if index % vis_frame_stride == 0:
                        cv2.imshow("Combined Image", cv2.resize(result, (640, 480)))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    new_images.append(result)
                del new_f[dataset_path]
                new_f.create_dataset(dataset_path, data=new_images)
                
    shutil.rmtree(jpg_dir)