import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

# 1️⃣ ResNet18 모델 로드 및 Feature Extractor 지정
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# ResNet18의 마지막 Convolution Layer 선택 (Feature Map 추출)
target_layer = model.layer4[-1]  # 마지막 Conv layer 사용 (feature map 제공)

# 2️⃣ 입력 이미지 전처리
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224) 형태로 변환

# image_path = "yolo/tmp_data/grasp_cable/image/00500.jpg"  # 사용할 이미지 경로

image_path = "yolo/raw_data/grasp_cable/011/00000.jpg"  # 사용할 이미지 경로
input_tensor = load_image(image_path) 

# 3️⃣ EigenCAM 적용
cam = EigenCAM(model, [target_layer])


# Classifier 없이 feature map 자체를 사용하기 때문에 targets=None 설정
grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # (1, 224, 224) 형태로 나옴
grayscale_cam = grayscale_cam[0]  # 배치 차원 제거


# 4️⃣ 원본 이미지 로드 및 Saliency Map 생성
original_image = np.array(Image.open(image_path).resize((224, 224))) / 255.0  # [0, 1] 범위로 정규화

visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

# 5️⃣ cv2를 사용한 시각화
visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식을 사용하므로 변환 필요

# 이미지 출력
cv2.imshow("Annotated Image", visualization_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()