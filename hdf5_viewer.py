import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def show_freq(img, title):
    # 2D FFT 수행
    R_fft_2d = np.fft.fft2(img[:, :, 0])
    G_fft_2d = np.fft.fft2(img[:, :, 1])
    B_fft_2d = np.fft.fft2(img[:, :, 2])

    # 주파수 성분의 진폭 계산
    R_magnitude_2d = np.abs(np.fft.fftshift(R_fft_2d))
    G_magnitude_2d = np.abs(np.fft.fftshift(G_fft_2d))
    B_magnitude_2d = np.abs(np.fft.fftshift(B_fft_2d))

    # 그래프로 표현
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.log(R_magnitude_2d + 1), cmap='gray')
    plt.title(f"{title} - 2D Frequency Spectrum of R")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(np.log(G_magnitude_2d + 1), cmap='gray')
    plt.title(f"{title} - 2D Frequency Spectrum of G")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.log(B_magnitude_2d + 1), cmap='gray')
    plt.title(f"{title} - 2D Frequency Spectrum of B")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def show_diff_img(img1, img2):
    # 두 이미지의 차이 계산
    difference = img1 - img2

    # 저주파 통과 필터 적용 (가우시안 블러)
    blurred_difference = cv2.GaussianBlur(difference, (15, 15), sigmaX=3)

    # 이미지 플롯
    plt.figure(figsize=(15, 5))

    # 차이 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(difference, cmap='gray', interpolation='nearest')
    plt.title('Original Difference Image')
    plt.colorbar()

    # 블러링 적용 후 차이 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_difference, cmap='gray', interpolation='nearest')
    plt.title('Blurred Difference Image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
        

def visualize_hdf5_images(hdf5_path, cameras=['camera1', 'camera2']):
    with h5py.File(hdf5_path, 'r') as f:
        images = []
        
        for im_name in cameras:
            images.append(f[f"observations/images/{im_name}"])

        episode_len = len(images[0])

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # 초기 이미지로 각 서브플롯 설정
        ims = []
        for i in range(2):
            ims.append(axs[i].imshow(images[i][0], cmap='gray'))
            axs[i].axis('off')  # 축 레이블 제거

        def update(frame):
            for i in range(2):
                current_image = images[i][frame][..., ::-1]
                ims[i].set_data(current_image)
            return ims
        
        ani = FuncAnimation(fig, update, frames=episode_len, blit=True, interval=50)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 기본값 설정
    dir = "./datasets"
    work = "grasp_cable"
    episode = "257"

    # HDF5 파일 경로
    hdf5_path = f"{dir}/{work}/original/episode_{episode}.hdf5"

    # 함수 호출
    visualize_hdf5_images(hdf5_path)