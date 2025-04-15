import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import cv2
import numpy as np

class CableDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, annotaion_type='xml', transform=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_filenames = [
            f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))
        ]
        self.annotation_type = annotaion_type

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        im_width = image.width
        im_height = image.height

        # XML 파일 로드
        boxes = []
        labels = []
        if self.annotation_type == 'xml':
            xml_filename = os.path.splitext(img_filename)[0] + ".xml"
            xml_path = os.path.join(self.annotations_dir, xml_filename)

            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name").text
                if label not in class_mapping:
                    continue  # 선택된 클래스가 아니면 건너뜀

                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_mapping[label])

            if not boxes:
                return None  # 선택한 클래스가 없으면 건너뜀
        elif self.annotation_type == 'txt':
            txt_filename = os.path.splitext(img_filename)[0] + ".txt"
            txt_path = os.path.join(self.annotations_dir, txt_filename)
            
            with open(txt_path, "r") as file:
                for line in file:
                    values = line.strip().split()  # 공백 기준으로 분리
                    label = int(values[0])  # 첫 번째 값은 정수 (클래스 ID)
                    if label not in class_mapping:
                        continue  # 선택된 클래스가 아니면 건너뜀
                    
                    x_center, y_center, width, height = map(float, values[1:])  # 나머지는 float 변환
                    xmin = (x_center - width/2) * im_width
                    xmax = (x_center + width/2) * im_width
                    ymin = (y_center - height/2) * im_height
                    ymax = (y_center + height/2) * im_height
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, {"boxes": boxes, "labels": labels}

# 변환 정의
transform = transforms.Compose([transforms.ToTensor()])


dataset_path = "yolo/tmp_data/grasp_cable/"
image_dir = os.path.join(dataset_path, "image")
annotations_dir = os.path.join(dataset_path, "label")
class_mapping = [0]
dataset = CableDataset(image_dir, annotations_dir, 'txt', transform)

# dataset_path = "kagglehub/datasets/andrewmvd/dog-and-cat-detection/versions/1"
# image_dir = os.path.join(dataset_path, "images")
# annotations_dir = os.path.join(dataset_path, "annotations")
# class_mapping = {"dog": 1, "cat": 2}
# dataset = CableDataset(image_dir, annotations_dir, 'xml', transform)


def visualize_dataset(data):
        # dataset 데이터
    image_tensor, target = data  # 첫 번째 데이터 가져오기
    boxes = target['boxes']  # 바운딩 박스 좌표
    labels = target['labels']  # 라벨

    # (C, H, W) -> (H, W, C) 변환 후 0~255로 변환
    image_np = image_tensor.numpy().transpose(1, 2, 0)  # PyTorch tensor -> numpy
    image_np = (image_np * 255).astype(np.uint8)  # float [0,1] -> int [0,255]

    # RGB -> BGR 변환 (OpenCV는 BGR 사용)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 박스 그리기
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)  # float -> int 변환
        cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # 빨간색 박스
        
        # 라벨 추가
        label_text = f"Class: {labels[0].item()}"  # 라벨을 정수로 변환
        cv2.putText(image_cv2, label_text, (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 이미지 출력
    cv2.imshow("Annotated Image", image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 빈 데이터 제거
filtered_dataset = [data for data in dataset if data is not None]
filtered_dataset = filtered_dataset[:300]
dataloader = DataLoader(filtered_dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

# Faster R-CNN 모델 정의
backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# backbone = torchvision.models.resnet18(weights=None)
backbone = nn.Sequential(*list(backbone.children())[:-2])  # 분류기 제거
backbone.out_channels = 512

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

model = FasterRCNN(backbone, num_classes=3, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

# 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 20

print('Training Start!')

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), 'cable_detector.pth')
print("Training Complete!")
