import os
import shutil
import random

def align_dataset(data_name):
    """
    데이터 폴더(data_name)와 라벨 폴더(data_name_labeled)에서
    파일 이름은 다르지만, 순서가 1:1 대응되어 있다고 가정할 때,
    train/valid/test 비율로 나누어 복사하는 함수.
    """

    # 1. 출력할 폴더 생성
    base_output_dir = os.path.join("data", data_name)

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
    data_dir = data_name
    label_dir = f"{data_name}_label"

    # 3. 파일 목록 불러오기 (정렬 후 리스트화)
    data_files = sorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
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
            src_data_path  = os.path.join(data_dir, data_file)
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
    print(f"Completed alignment for {data_name}:")
    print(f" - Train: {len(train_indices)} pairs")
    print(f" - Valid: {len(valid_indices)} pairs")
    print(f" - Test:  {len(test_indices)} pairs")

if __name__ == "__main__":
    # 사용 예시
    align_dataset("tmp")