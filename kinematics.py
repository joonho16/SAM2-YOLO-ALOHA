import torch

# cuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    )
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

class Kinematics:
    def __init__(self, kin_config):
        # kin_config = {
        #     'robot_file': 'ur5e.yml',
        #     'world_file': "collision_base.yml",
        #     'rotation_threshold': 0.1,
        #     'position_threshold': 0.01,
        #     'num_seeds': 1,
        #     'self_collision_check': True,
        #     'self_collision_opt': False,
        #     'tensor_args': TensorDeviceType(),
        #     'use_cuda_graph': True
        # }
        
        self.kin_config = kin_config
        
        self.robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), kin_config['robot_file']))['robot_cfg']
        )
        self.world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), kin_config['world_file'])))
        
        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            rotation_threshold=kin_config['rotation_threshold'],
            position_threshold=kin_config['position_threshold'],
            num_seeds=kin_config['num_seeds'],
            self_collision_check=kin_config['self_collision_check'],
            self_collision_opt=kin_config['self_collision_opt'],
            tensor_args=kin_config['tensor_args'],
            use_cuda_graph=kin_config['use_cuda_graph'],
        )
        
        self.ik_solver = IKSolver(ik_config)
        
        self.tensor_args = TensorDeviceType()
        
    def forward_kinematics(self, qpos):
        qpos = torch.tensor(qpos, requires_grad=True, **(self.tensor_args.as_torch_dict()))
        raw_xpos = self.ik_solver.fk(qpos)
        xpos = torch.cat((raw_xpos.ee_position, raw_xpos.ee_quaternion), dim=1)
        xpos = xpos.squeeze(0)
        return xpos
    
    def inverse_kinematics(self, xpos, q_init):

        xpos = torch.tensor(xpos, **(self.tensor_args.as_torch_dict()))
        q_init = torch.tensor(q_init, **(self.tensor_args.as_torch_dict()))
        
        ee_position = xpos[:3].unsqueeze(0)
        ee_quaternion = xpos[3:].unsqueeze(0)
        goal = Pose(ee_position, ee_quaternion)
        q_init = q_init.unsqueeze(0)
        
        # debugpy.breakpoint()
        
        # result = self.ik_solver.solve_single(
        #     goal_pose = goal,
        #     seed_config = q_init,
        # )

        result = self.ik_solver.solve_single(
            goal_pose=goal,
            retract_config=q_init.reshape(-1),
            seed_config=q_init.reshape(1, 1, -1),
            return_seeds=1,
            num_seeds=1000,
            use_nn_seed=False,
            newton_iters=50,
        )
        
        q_solution = result.solution[result.success]
        q_solution = q_solution.squeeze(0)
        return q_solution