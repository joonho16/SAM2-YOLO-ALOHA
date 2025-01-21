import sys
import os
import hydra
import h5py
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
import sam2
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
config_dir = "sam2/sam2/configs/sam2.1"
model_cfg = "sam2.1_hiera_l.yaml"

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def calculate_bounding_box(outmask):
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

def save_yolo_labels(bboxes, image_width, image_height, output_file, class_id=0):

    with open(output_file, 'w') as file:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            # Calculate YOLO format values
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Write to file
            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main(video_dir, output_dir, points, labels, class_id):

    # Hydra 설정 경로 초기화
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # 기존 설정 초기화
    hydra.initialize(config_path=config_dir, version_base=None)

    # 모델 생성 및 예측
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    np.random.seed(3)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    # 첫번째 프레임 띄워서 포인트 프롬프트 저장
    frame_idx = 0
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    plt.show()
    quit()


    ann_frame_idx = 0

    for (index, point) in enumerate(points):
        # Let's add a positive click at (x, y) = (210, 350) to get started
        point = np.array([point], dtype=np.float32)
        real_labels = np.array([labels[index]], np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=index,
            points=point,
            labels=real_labels,
        )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    # 비디오로 세그먼트
    vis_frame_stride = 300
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names)):

        image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        width, height = image.size
        if out_frame_idx % vis_frame_stride == 0:
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(image)
        boxes = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            box = calculate_bounding_box(out_mask)
            if box is None:
                continue
            else:
                x_min, y_min, x_max, y_max = box
            boxes.append([x_min, y_min, x_max, y_max])
            if out_frame_idx % vis_frame_stride == 0:
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        output_file = os.path.join(output_dir, f"{out_frame_idx:05}.txt")
        save_yolo_labels(boxes, width, height, output_file, class_id)

    plt.show()



if __name__ == '__main__':

    video_dir = "yolo/raw_data/gripper"
    output_dir = "yolo/raw_data/gripper_label/"
    points = [[[71, 110],[73, 80]]]
    labels = [[1, 1]]
    # points = [[[22, 38]], [[53, 14]], [[118, 30]], [[127, 39]], [[137, 79]], [[122, 85]], [[135, 100]], [[82, 110]], [[51, 113]], [[49, 148]]]
    # labels = ([1], [1], [1], [1], [1], [1], [1], [1], [1], [1])
    class_id = 1
    main(video_dir, output_dir, points, labels, class_id)