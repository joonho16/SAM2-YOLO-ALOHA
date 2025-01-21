import cv2
import os

def save_frames_with_lower_resolution(mp4_path, output_dir, img_size):
    """
    MP4 동영상을 프레임 단위로 추출하여,
    해상도를 낮춘 뒤(JPG)로 저장하는 함수.

    Parameters
    ----------
    mp4_path : str
        읽어올 MP4 파일 경로
    output_dir : str
        프레임 이미지를 저장할 디렉토리 경로
    scale : float
        해상도를 낮추는 비율 (0 < scale <= 1)
        예) scale=0.5 라면 가로/세로 해상도를 절반으로 줄임
    """
    # 저장 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 동영상 불러오기
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print("동영상을 열 수 없습니다.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # 해상도 낮추기
        new_width = img_size[0]
        new_height = img_size[1]
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # 파일 이름 정하기
        file_name = os.path.join(output_dir, f"{frame_count:05d}.jpg")

        # 이미지 저장 (JPG 형식)
        cv2.imwrite(file_name, frame_resized)

        frame_count += 1

    cap.release()
    print(f"총 {frame_count}개의 프레임이 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    # 예시 사용
    input_video_path = "gripper2.mp4"         # mp4 파일 경로
    output_directory = "raw_data/gripper2"     # 저장 폴더
    # scale_ratio = 0.5                        # 해상도 축소 비율(50%로 축소)
    img_size = (160, 160)

    save_frames_with_lower_resolution(input_video_path, output_directory, img_size)