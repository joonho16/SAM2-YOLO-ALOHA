import os
import glob
import cv2
import numpy as np

# imgaug
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# ===================================
# 폴더 설정
# ===================================
IMG_DIR = 'tmp'
LABEL_DIR = 'tmp_label'

# ===================================
# YOLO <-> VOC 변환 함수
# ===================================
def yolo_to_voc(x_center, y_center, w, h, img_w, img_h):
    """
    YOLO 정규화 좌표 -> PASCAL VOC 픽셀 좌표
    (x_min, y_min, x_max, y_max)
    """
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h
    w_abs = w * img_w
    h_abs = h * img_h

    x_min = x_center_abs - (w_abs / 2.0)
    y_min = y_center_abs - (h_abs / 2.0)
    x_max = x_center_abs + (w_abs / 2.0)
    y_max = y_center_abs + (h_abs / 2.0)
    return x_min, y_min, x_max, y_max

def voc_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    """
    PASCAL VOC 픽셀 좌표 -> YOLO 정규화 좌표
    (class_id, x_center, y_center, w, h)
    """
    w_box = x_max - x_min
    h_box = y_max - y_min
    x_center = x_min + w_box / 2.0
    y_center = y_min + h_box / 2.0

    # 정규화 (0~1)
    x_center /= img_w
    y_center /= img_h
    w_box /= img_w
    h_box /= img_h
    return x_center, y_center, w_box, h_box

def main(img_dir, label_dir):
    # ===================================
    # 증강 파이프라인 (개별 적용)
    # ===================================
    rotate_90_right = iaa.Rotate(90)       # 오른쪽 90도 회전
    rotate_90_left  = iaa.Rotate(-90)      # 왼쪽 90도 회전
    rotate_180      = iaa.Rotate(180)      # 180도 회전
    brightness      = iaa.Multiply((0.5, 1.5))  # 밝기 조절 (0.5배 ~ 1.5배)

    # 어떤 증강에 어떤 번호를 붙일지 매핑(1~4)
    AUGMENTATIONS = [
        (rotate_90_right, 1),
        (rotate_90_left,  2),
        (rotate_180,      3),
        (brightness,      4),
    ]

    # tmp 폴더 내 모든 .jpg 파일 목록
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)      # 예: '00001.jpg'
        name_no_ext = os.path.splitext(img_name)[0]  # '00001'

        # 해당 이미지 번호를 파싱
        # (ex. '00001' -> 1)
        try:
            original_num = int(name_no_ext)
        except ValueError:
            # 만약 숫자 형식이 아니면 스킵
            print(f"Warning: 파일명이 숫자가 아님: {img_name}")
            continue

        # 라벨(txt) 경로
        label_path = os.path.join(label_dir, f'{name_no_ext}.txt')
        if not os.path.exists(label_path):
            print(f"Warning: 라벨 파일이 없습니다: {label_path}")
            continue

        # 이미지, 라벨 로드
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: 이미지 로드 실패: {img_path}")
            continue
        img_h, img_w = image.shape[:2]

        with open(label_path, 'r') as f:
            lines = f.read().strip().split('\n')

        bboxes = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            class_id = parts[0]
            x_center, y_center, w_box, h_box = map(float, parts[1:5])
            x_min, y_min, x_max, y_max = yolo_to_voc(x_center, y_center, w_box, h_box, img_w, img_h)
            bboxes.append(BoundingBox(
                x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id
            ))
        bbs_on_image = BoundingBoxesOnImage(bboxes, shape=image.shape)

        # 4가지 증강 각각 적용
        for augmenter, aug_idx in AUGMENTATIONS:
            aug_result = augmenter(image=image, bounding_boxes=bbs_on_image)
            aug_img, aug_bbs = aug_result[0], aug_result[1]

            # 증강된 이미지의 높이/너비
            aug_h, aug_w = aug_img.shape[:2]

            # 새 파일 번호 = (원본번호 * 10) + 증강ID
            # 예) 원본 '00001' -> int 1, 증강ID 3 -> 13 -> '00013'
            new_num = original_num * 10 + aug_idx

            # 원본과 같은 자릿수로 zero-padding
            # ex) len('00001')=5 → "{13:05d}" -> '00013'
            zero_padded = f"{new_num:0{len(name_no_ext)}d}"

            # 새 파일명
            new_img_name = zero_padded + '.jpg'
            new_label_name = zero_padded + '.txt'

            # 바운딩박스 YOLO 좌표로 되돌려 저장
            new_label_path = os.path.join(label_dir, new_label_name)
            with open(new_label_path, 'w') as f_out:
                for bb in aug_bbs:
                    # 이미지 범위를 벗어난 박스를 clip
                    bb = bb.clip_out_of_image((aug_h, aug_w))
                    if bb.area <= 0:
                        # 완전히 잘려나간 경우 스킵
                        continue

                    x_min, y_min, x_max, y_max = bb.x1, bb.y1, bb.x2, bb.y2
                    class_id = bb.label  # 원본 class_id
                    x_c, y_c, w_b, h_b = voc_to_yolo(x_min, y_min, x_max, y_max, aug_w, aug_h)

                    # 혹시 0~1 범위를 벗어날 수 있으므로 clip
                    x_c = max(min(x_c, 1.0), 0.0)
                    y_c = max(min(y_c, 1.0), 0.0)
                    w_b = max(min(w_b, 1.0), 0.0)
                    h_b = max(min(h_b, 1.0), 0.0)

                    f_out.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_b:.6f} {h_b:.6f}\n")

            # 증강된 이미지 저장
            new_img_path = os.path.join(img_dir, new_img_name)
            cv2.imwrite(new_img_path, aug_img)
            print(f"Saved: {new_img_path}, {new_label_path}")

if __name__ == "__main__":
    main(IMG_DIR, LABEL_DIR)