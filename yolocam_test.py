from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np
from utils import mask_outside_boxes

yolo_model = YOLO('runs/detect/train11/weights/best.pt')
backbone_layers = nn.Sequential(*list(yolo_model.model.model.children())[:8])

class YOLOBackbone(nn.Module):
    def __init__(self, backbone):
        super(YOLOBackbone, self).__init__()
        self.backbone = backbone
    
    def forward(self, x):
        return self.backbone(x)

backbone_model = YOLOBackbone(backbone_layers)

# GPU 사용 가능하면 GPU로 모델 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_model.to(device)
backbone_model.eval()  # 모델을 평가 모드로 설정

image = cv2.imread("yolo/tmp_data/grasp_cable/image/00500.jpg")
image = cv2.imread("yolo/raw_data/grasp_cable/010/00030.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 정규화 및 차원 변경 (B, C, H, W 형식으로 변환)
image_tensor = torch.from_numpy(image).float() / 255.0  # [0,1]로 정규화
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) → (1, C, H, W)
image_tensor = image_tensor.to(device)  # GPU 사용 가능하면 이동

# Backbone 모델을 통과시켜 feature 추출
with torch.no_grad():
    feature_map = backbone_model(image_tensor)
    print(feature_map.size())

target_layer = backbone_model.backbone[-1]

cam = EigenCAM(model=backbone_model, target_layers=[target_layer])

# EigenCAM 결과 얻기
grayscale_cam = cam(input_tensor=image_tensor)  # CAM 결과 (20x20 크기 예상)
grayscale_cam = grayscale_cam[0]  # 첫 번째 배치만 사용

heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

original_image = np.array(image) / 255.0  # [0, 1] 범위로 정규화
visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

# 5️⃣ cv2를 사용한 시각화
visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식을 사용하므로 변환 필요

# 이미지 출력
cv2.imshow("Annotated Image", visualization_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

yolo_output = yolo_model(image)
print(yolo_output[0].boxes)
masked_image = mask_outside_boxes(image, yolo_output[0].boxes)

# 이미지 출력
cv2.imshow("Annotated Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

