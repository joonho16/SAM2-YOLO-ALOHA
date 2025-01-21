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

checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
config_dir = "sam2/sam2/configs/sam2.1"
model_cfg = "sam2.1_hiera_t.yaml"

# Hydra 설정 경로 초기화
hydra.core.global_hydra.GlobalHydra.instance().clear()  # 기존 설정 초기화
hydra.initialize(config_path=config_dir, version_base=None)

# 모델 생성 및 예측
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 기본값 설정
dir = "./datasets"
work = "grasp_cable"
episode = "2"

# HDF5 파일 경로
hdf5_path = f"{dir}/{work}/episode_{episode}.hdf5"

np.random.seed(3)

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


# # hdf5 사용하기      
# with h5py.File(hdf5_path, 'r') as f:
#     images = []
    
#     images = np.stack([f[f"observations/images/{im_name}"][:] for im_name in ['camera1', 'camera2']])
#     images = np.transpose(images[0], (0, 3, 1, 2))
#     images[0] = images[100]

#     images = images[:5]

# inference_state = predictor.init_state(video_array=images)
    


# video url 사용하기
# video_dir = os.path.join("..", "sam2/demo/data/gallery/jpeg_gallery")
video_dir = "yolo/gripper"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)




predictor.reset_state(inference_state)

ann_frame_idx = [0, 50]  # the frame index we interact with
ann_obj_id = [0, 1]  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[106, 68]], dtype=np.float32)
box = np.array([92, 0, 108, 82], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx[0],
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[120, 60]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx[0],
    obj_id=ann_obj_id[1],
    points=points,
    labels=labels,
)



# points = np.array([[130, 58]], dtype=np.float32)
# labels = np.array([1], np.int32)

# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx[1],
#     obj_id=ann_obj_id[1],
#     points=points,
#     labels=labels,
# )

# show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(images[0].transpose(1,2,0))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# # hdf5로 세그먼트
# vis_frame_stride = 10
# plt.close("all")
# for out_frame_idx in range(0, images.shape[0], vis_frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(images[out_frame_idx].transpose(1,2,0))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    
# 비디오로 세그먼트
vis_frame_stride = 10
plt.close("all")
for out_frame_idx in range(0, 10, vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

plt.show()