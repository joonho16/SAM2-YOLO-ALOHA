import h5py
import numpy as np
import plotly.graph_objects as go
import os
from curobo.types.base import TensorDeviceType

import kinematics


from constants import TASK_CONFIGS


def plot_pos_val(hdf5_dir, task_config, task_space):
    kn = kinematics.Kinematics(kin_config)
    file_names = os.listdir(hdf5_dir)
    fig = go.Figure()
    vector_sequences = {}
    for i, file_name in enumerate(file_names):
        hdf5_path = hdf5_dir + '/' + file_name
        with h5py.File(hdf5_path, 'r') as f:
            actions = f[f"action"][:]
            xactions = f[f"xaction"][:]
            xpos_data = f["observations/xpos"][:]
            qpos_data = f["observations/qpos"][:]
            print(xpos_data[0])
            print(qpos_data[0])
            
            vectors = np.array(actions)

            vector_sequences[file_name] = vectors
            x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],  # 첫 번째 점만 표시
                mode='lines+markers',
                marker=dict(size=3),
                line=dict(width=2),
                name=f"{file_name}",
                visible=True  # 초기에는 모두 보이게
            ))
            
            # vectors_fk = []
            # for action in actions:
            #     action_fk = kn.forward_kinematics(action[:-1]).detach().cpu().numpy()
            #     vectors_fk.append(action_fk)

            # vectors_fk = np.array(xactions)
            # vector_sequences['xact_' + file_name] = vectors_fk
            # x, y, z = vectors_fk[:, 0], vectors_fk[:, 1], vectors_fk[:, 2]
            # fig.add_trace(go.Scatter3d(
            #     x=[x[0]], y=[y[0]], z=[z[0]],  # 첫 번째 점만 표시
            #     mode='lines+markers',
            #     marker=dict(size=3),
            #     line=dict(width=2),
            #     name=f"{file_name}",
            #     visible=True  # 초기에는 모두 보이게
            # ))
            
    frames = []
    for t in range(100):
        frame_data = []
        for file_name, vectors in vector_sequences.items():
            vectors = np.array(vectors)
            x, y, z = vectors[:t+1, 0], vectors[:t+1, 1], vectors[:t+1, 2]

            frame_data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                marker=dict(size=3),
                line=dict(width=2),
                name=f"{file_name}",
            ))
        
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # 애니메이션 업데이트 설정
    fig.update(frames=frames)
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title="3D Vector Sequence Animation",
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                 "label": "▶ Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
                                   "transition": {"duration": 0}}],
                 "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

        
if __name__ == "__main__":
    # 기본값 설정
    dir = "./datasets"
    work = "grasp_cable"
    episode = "0"
    task_config = TASK_CONFIGS[work]
    
    kin_config = {
        'robot_file': 'ur5e.yml',
        'world_file': "collision_base.yml",
        'rotation_threshold': 0.1,
        'position_threshold': 0.01,
        'num_seeds': 1,
        'self_collision_check': True,
        'self_collision_opt': False,
        'tensor_args': TensorDeviceType(),
        'use_cuda_graph': True
    }

    # HDF5 파일 경로
    hdf5_dir = f"{task_config['dataset_dir']}/debug_hc_1"
    
    task_space = True
    # 함수 호출
    # replay_episode(hdf5_path, task_config, kin_config, task_space)
    plot_pos_val(hdf5_dir, task_config, task_space)

