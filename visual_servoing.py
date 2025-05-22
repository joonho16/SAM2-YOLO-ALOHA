from ultralytics import YOLO


from env import AlohaEnv
import rospy

from curobo.types.base import TensorDeviceType

import cv2
import time

import numpy as np

from PIL import Image


if __name__ == '__main__':
    rospy.init_node("visual_servoing_node", anonymous=False)
    yolo_model = YOLO('yolo/tomato.pt')

    kin_config = {
        'robot_file': 'ur5e.yml',
        'world_file': "collision_base.yml",
        'rotation_threshold': 0.01,
        'position_threshold': 0.01,
        'num_seeds': 500,
        'self_collision_check': False,
        'self_collision_opt': False,
        'tensor_args': TensorDeviceType(),
        'use_cuda_graph': True
    }
    env = AlohaEnv(kin_config, ['camera1', 'camera2'])

    rate = rospy.Rate(30)  # 초당 30회 루프 실행


    home_pos = [-1.6939677000045776, -1.266764299278595, -1.2382834593402308, -0.3451989454082032, 1.3663817644119263, -1.4772747198687952, 0.087]
    # home_pos = [-1.5704529285430908, -1.2528355580619355, -0.9905026594745081, -0.4378644985011597, 0.9186715483665466, -1.4332850615130823,  0.087]
    env.move_joint(home_pos)
    time.sleep(3)

    current_qpos = env.get_qpos()
    target_qpos = list(current_qpos)
    target_qpos[5] = 0
    # env.move_joint(target_qpos)

    state = 0   # 0: tomato to center / 1: rotate wrist / 2: approach to tomato

    counter = 0
    img_list = []
    while not rospy.is_shutdown():
        if env.camera2 is not None:
            img = env.camera2.copy()
            width = img.shape[1]
            height = img.shape[0]
            yolo_result = yolo_model(img)[0]
            boxes = yolo_result.boxes
            target_index_list = np.where(boxes.cls.cpu().numpy() == 0)[0]
            if len(target_index_list) > 0:
                target_index = target_index_list[0]
                xyxy = boxes.xyxy.cpu().numpy()[target_index]
                xywh = boxes.xywh.cpu().numpy()[target_index]
                p1 = (int(xyxy[0]), int(xyxy[1]))
                p2 = (int(xyxy[2]), int(xyxy[3]))
                c = (int(xywh[0]), int(xywh[1]))
                cv2.rectangle(img, p1, p2, (0, 255, 0), 4)
                cv2.circle(img, c, 3, (0, 0, 255), 3)
                
                current_xpos = env.get_xpos()
                current_qpos = env.get_qpos()
                target_xpos = list(current_xpos)

                if state == 0:
                    scale = 0.02
                    center_range_x = (420, 440)
                    center_range_y = (160, 180)
                    cv2.rectangle(img, (center_range_x[0], center_range_y[0]), (center_range_x[1], center_range_y[1]), (0, 0, 255), 4)
                    
                    completed = True
                    if c[0] < center_range_x[0]:
                        target_xpos[0] -= scale
                        completed = False
                    elif c[0] > center_range_x[1]:
                        target_xpos[0] += scale
                        completed = False
                    if c[1] < center_range_y[0]:
                        target_xpos[2] += scale
                        completed = False
                    elif c[1] > center_range_y[1]:
                        target_xpos[2] -= scale
                        completed = False
                    if completed:
                        state = 1

                    env.move_xstep(target_xpos)

                elif state == 1:
                    state = 2
                    # target_qpos = list(current_qpos)
                    # target_qpos[5] = 0
                    # env.move_joint(target_qpos)
                    # state = 2

                elif state == 2:
                    counter += 1
                    target_xpos[1] += 0.02
                    size = p2[0] - p1[0]
                    target_size = 260

                    if size > target_size:
                        state = 3
                    env.move_xstep(target_xpos)
                    counter += 1
                    if counter > 10:
                        state = 0
                        counter = 0
                
                elif state == 3:
                    state = 4
                    # scale = 0.04
                    # center_range_x = (0, 640)
                    # center_range_y = (160, 200)
                    # cv2.rectangle(img, (center_range_x[0], center_range_y[0]), (center_range_x[1], center_range_y[1]), (0, 0, 255), 4)
                    # completed = True
                    # if c[1] < center_range_y[0]:
                    #     target_xpos[0] += scale
                    #     completed = False
                    # elif c[1] > center_range_y[1]:
                    #     target_xpos[0] -= scale
                    #     completed = False
                    # if completed:
                    #     state = 4
                    # env.move_xstep(target_xpos)

                elif state == 4:
                    target_xpos[7] = 0
                    if current_qpos[6] < 0.005:
                        state = 5
                    env.move_xstep(target_xpos)

                elif state == 5:
                    target_xpos[1] -= 0.05
                    counter += 1
                    if counter > 30:
                        counter = 0
                        state = 6
                    env.move_joint(home_pos)


            img_list.append(img)    
            cv2.imshow('camera2', img)
        else:
            print('No Image')
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        rate.sleep()

    cv2.destroyAllWindows()

    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in img_list]

    pil_images[0].save(
        'output2.gif',
        save_all=True,
        append_images=pil_images[1:],
        duration=50,  # 프레임 간 시간 (ms)
        loop=0         # 0: 무한 반복
    )

