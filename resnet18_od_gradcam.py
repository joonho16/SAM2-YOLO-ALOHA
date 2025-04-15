import os
import torch
import xml.etree.ElementTree as ET
from torchvision.transforms import ToPILImage
from torchvision.models.detection import FasterRCNN
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
import cv2
import numpy as np


# 클래스 매핑
class_mapping = [0]

# 변환 정의
transform = transforms.Compose([transforms.ToTensor()])

# Faster R-CNN 모델 정의
backbone = torchvision.models.resnet18(weights=None)
backbone = nn.Sequential(*list(backbone.children())[:-2])  # 분류기 제거
backbone.out_channels = 512

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

model = FasterRCNN(backbone, num_classes=3, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("cable_detector.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.to(device)
model.eval()  # 평가 모드 설정

target_layer = model.backbone[-1]

# 3️⃣ EigenCAM 적용
cam = EigenCAM(model.backbone, [target_layer])

# 2️⃣ 입력 이미지 전처리
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

# 데이터셋 첫 번째 샘플 가져오기
# image_path = "kagglehub/datasets/andrewmvd/dog-and-cat-detection/versions/1/images/Cats_Test800.png"  # 사용할 이미지 경로
image_path = "yolo/tmp_data/grasp_cable/image/00500.jpg"

input_tensor = load_image(image_path)


# Classifier 없이 feature map 자체를 사용하기 때문에 targets=None 설정
grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # (1, 224, 224) 형태로 나옴
grayscale_cam = grayscale_cam[0]  # 배치 차원 제거

# 4️⃣ 원본 이미지 로드 및 Saliency Map 생성
original_image = np.array(Image.open(image_path)) / 255.0  # [0, 1] 범위로 정규화
visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

# # 5️⃣ cv2를 사용한 시각화
visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식을 사용하므로 변환 필요

# 이미지 출력
cv2.imshow("Annotated Image", visualization_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_tensor = torch.tensor(np.transpose(original_image, (2, 0, 1))).to(dtype=torch.float32).to(device)

# 모델 예측
with torch.no_grad():
    prediction = model([image_tensor])

# 결과 추출
pred_boxes = prediction[0]["boxes"].cpu().numpy()
pred_labels = prediction[0]["labels"].cpu().numpy()
pred_scores = prediction[0]["scores"].cpu().numpy()

# 바운딩 박스가 일정 확률 이상인 것만 선택 (예: 0.5 이상)
threshold = 0.5
filtered_boxes = pred_boxes[pred_scores >= threshold]
filtered_labels = pred_labels[pred_scores >= threshold]

# 클래스 매핑 (숫자 → 문자열)
image_cv2 = np.array(image_tensor.cpu())
image_cv2 = np.transpose(image_cv2, (1, 2, 0))
image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식 사용

for i, box in enumerate(filtered_boxes):
    xmin, ymin, xmax, ymax = map(int, box)  # float -> int 변환
    label = filtered_labels[i]
    
    # 바운딩 박스 그리기
    cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # 빨간색 (BGR)

    # 라벨 텍스트 추가
    text = f"{label} ({pred_scores[i]:.2f})"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image_cv2, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), (255, 255, 255), -1)  # 배경
    cv2.putText(image_cv2, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 이미지 표시
cv2.imshow("Result", image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()