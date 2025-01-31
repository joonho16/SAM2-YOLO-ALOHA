import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def augment_data(hdf5_path, new_hdf5_path, darken_factor, cameras):
    with h5py.File(hdf5_path, 'r') as f:
        with h5py.File(new_hdf5_path, 'w') as new_f:
            # 전체 그룹 구조를 복사 (root 그룹 복사)
            for key in f.keys():
                f.copy(key, new_f, name=key)  # 소스와 대상 그룹 이름 동일하게 설정

            for im_name in cameras:
                # 원본 데이터 불러오기
                dataset_path = f"observations/images/{im_name}"
                images = np.array(f[dataset_path])  # Dataset -> NumPy 배열

                # 밝기 조정
                darker_images = np.clip(images * darken_factor, 0, 255).astype(np.uint8)

                # 기존 데이터 대체
                del new_f[dataset_path]
                new_f.create_dataset(dataset_path, data=darker_images)


if __name__ == "__main__":
    # 기본값 설정
    dir = "./datasets"
    work = "grasp_cable"
    
    # folders = ['original', 'hgdagger']
    folders = ['original']
    d_count = 0
    for folder in folders:
        data_dir = f"{dir}/{work}/{folder}"
        data_len = len(os.listdir(data_dir))

        for i in range(data_len):
            hdf5_path = f"{data_dir}/episode_{i}.hdf5"

            new_dir = f"{dir}/{work}/aug"
            os.makedirs(new_dir, exist_ok=True)
            new_hdf5_path = f"{new_dir}/episode_{d_count}.hdf5"

            # 다크한 이미지로 변환하여 저장
            augment_data(hdf5_path, new_hdf5_path, 0.7, cameras=['camera1', 'camera2'])
            d_count += 1
            print(f"{d_count}번 에피소드가 저장되었습니다.")

            # 밝은 이미지로 변환하여 저장
            new_hdf5_path = f"{new_dir}/episode_{d_count}.hdf5"
            augment_data(hdf5_path, new_hdf5_path, 1.3, cameras=['camera1', 'camera2'])
            d_count += 1
            print(f"{d_count}번 에피소드가 저장되었습니다.")