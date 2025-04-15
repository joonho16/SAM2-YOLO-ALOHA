import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import numpy as np
import os
import glob

from utils import zoom_image, ros_image_to_numpy
from constants import TASK_CONFIGS

class ImageSubscriber:
    def __init__(self, topic_name):
        self.bridge = CvBridge()
        self.cam_name = topic_name
        self.image = None
        self.lock = threading.Lock()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.callback)

    def callback(self, data):
        try:
            cv_image = ros_image_to_numpy(data)
            with self.lock:
                self.image = cv_image
        except CvBridgeError as e:
            rospy.logerr(e)


def main(camera_names, output_dir, task_name):
    """
    두 카메라를 동시에 구독하고, 스페이스바로 녹화 on/off를 토글하여 mp4 파일 저장.
    """
    rospy.init_node('image_subscriber', anonymous=True)
    os.makedirs(output_dir, exist_ok=True)

    # 카메라별 subscriber 생성
    subscribers = {}
    for cam in camera_names:
        topic = f'/{cam}/color/image_raw/compressed'
        subscribers[cam] = ImageSubscriber(topic)

    # TASK_CONFIGS에서 카메라별 설정 불러오기
    camera_config = TASK_CONFIGS[task_name]['camera_config']

    # 비디오 파일 관련 변수
    is_recording = False
    video_writers = {}   # 각 카메라별 VideoWriter
    video_count = 0      # mp4 파일 번호

    # 루프 주기 (초당 프레임)
    # 녹화 프레임레이트도 아래 값과 맞추면 됩니다. (코드 하단의 cv2.VideoWriter_fourcc 설정 참조)
    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        # 각 카메라의 최신 이미지를 확인
        for cam in camera_names:
            image = subscribers[cam].image
            if image is None:
                # 아직 해당 카메라에서 이미지가 들어오지 않았으면 스킵
                continue
            
            # 카메라별 config(zoom, resize 등) 적용
            if cam in camera_config:
                if 'zoom' in camera_config[cam]:
                    size = camera_config[cam]['zoom']['size']
                    point = camera_config[cam]['zoom']['point']
                    image = zoom_image(image, point, size)
                if 'resize' in camera_config[cam]:
                    new_size = camera_config[cam]['resize']['size']
                    image = cv2.resize(image, new_size)
            
            # 보기 편하게 표시용으로 축소(640x480)
            max_height = 480
            max_width = 640
            preview_img = cv2.resize(image, (max_width, max_height))

            # 카메라별로 다른 윈도우 이름으로 보여주기
            window_name = f"Preview_{cam}"
            cv2.imshow(window_name, preview_img)

            # 녹화 중이면 프레임 작성
            if is_recording and cam in video_writers and video_writers[cam] is not None:
                video_writers[cam].write(image)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 'q' 키로 종료
            break
        elif key == ord(' '):
            # 스페이스바로 녹화 on/off 토글
            is_recording = not is_recording
            if is_recording:
                # 새 mp4 파일 오픈 (각 카메라별)
                for cam in camera_names:
                    image = subscribers[cam].image
                    if image is not None:
                        # 파일명: camera1_video_00000.mp4 이런 식
                        video_filename = f"{cam}_video_{video_count:05d}.mp4"
                        save_path = os.path.join(output_dir, video_filename)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # (width, height) = (image.shape[1], image.shape[0])
                        video_writers[cam] = cv2.VideoWriter(
                            save_path, fourcc, 10, (image.shape[1], image.shape[0])
                        )
                        rospy.loginfo(f"Start recording {save_path}")
                    else:
                        video_writers[cam] = None
            else:
                # 녹화 중지, 영상 저장 종료
                for cam in camera_names:
                    if video_writers.get(cam) is not None:
                        video_writers[cam].release()
                        video_writers[cam] = None
                rospy.loginfo("Stop recording.")
                video_count += 1

        rate.sleep()

    # 종료 시 리소스 해제
    for cam in camera_names:
        if video_writers.get(cam) is not None:
            video_writers[cam].release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_names = ["camera1", "camera2"]
    output_dir = "yolo/raw_data/tomato_data"
    task_name = "pick_tomato"
    main(camera_names, output_dir, task_name)