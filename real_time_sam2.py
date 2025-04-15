import pyrealsense2 as rs
import numpy as np
import torch
from sam2_real_time.sam2.build_sam import build_sam2_camera_predictor
import cv2

checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)     # depth 스트림 (선택)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # color 스트림

# 파이프라인 시작
pipeline.start(config)

if_init = False

try:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            # 파이프라인으로부터 프레임 수신
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # numpy array로 변환
            frame = np.asanyarray(color_frame.get_data())

            # 이후 logic은 웹캠 처리와 동일
            if not if_init:
                predictor.load_first_frame(frame)
                if_init = True
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=0
                )
            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)
                # ... 추론 결과 처리

            # 시각화나 exit 조건 등을 여기에 추가
            # 예: ESC 누르면 빠져나오기
            cv2.imshow('RealSense Stream', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

except KeyboardInterrupt:
    pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()