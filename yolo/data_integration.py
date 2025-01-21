import os
import shutil

# 폴더 경로 설정
tomatos_folder = "raw_data/tomatos"
gripper_folder = "raw_data/gripper"
tomatos_labeled_folder = "raw_data/tomatos_label"
gripper_labeled_folder = "raw_data/gripper_label"

tmp_folder = "tmp"
tmp_labeled_folder = "tmp_label"

# 병합 폴더 생성
os.makedirs(tmp_folder, exist_ok=True)
os.makedirs(tmp_labeled_folder, exist_ok=True)

# 파일 병합 함수
def merge_folders(source_folder, source_labeled_folder, target_folder, target_labeled_folder, start_index):
    count = start_index
    for filename in sorted(os.listdir(source_folder)):
        if filename.endswith(".jpg"):
            # 이미지 파일 복사
            src_image_path = os.path.join(source_folder, filename)
            dst_image_path = os.path.join(target_folder, f"{count:05}.jpg")
            shutil.copy(src_image_path, dst_image_path)
            
            # 라벨 파일 복사
            label_filename = filename.replace(".jpg", ".txt")
            src_label_path = os.path.join(source_labeled_folder, label_filename)
            dst_label_path = os.path.join(target_labeled_folder, f"{count:05}.txt")
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
            
            count += 1
    return count

# 병합 실행
next_index = merge_folders(tomatos_folder, tomatos_labeled_folder, tmp_folder, tmp_labeled_folder, 0)
merge_folders(gripper_folder, gripper_labeled_folder, tmp_folder, tmp_labeled_folder, next_index)

print("병합 완료!")