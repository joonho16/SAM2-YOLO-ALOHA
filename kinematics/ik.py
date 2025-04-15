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

tensor_args = TensorDeviceType()
world_file = "collision_base.yml"

robot_file = "ur5e.yml"
robot_cfg = RobotConfig.from_dict(
    load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
)
world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    world_cfg,
    rotation_threshold=0.01,
    position_threshold=0.01,
    num_seeds=1,
    self_collision_check=True,
    self_collision_opt=False,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)

for i in range(10):
    q_init = torch.tensor(
        [[-1.954, -1.108, -1.117, 0.158, 1.472, -1.511]],
        **(tensor_args.as_torch_dict())
    )

    # q_sample = ik_solver.sample_configs(1)
    q_sample = torch.tensor([-1.954, -1.108, -1.117, 0.158, 1.472, -1.511], **(tensor_args.as_torch_dict()))
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    
    print(q_init)
    print(goal)

    result = ik_solver.solve_single(
        goal_pose = goal,
        seed_config = q_init
    )

    q_solution = result.solution[result.success]

    # print(f"q_sample: {q_sample}")
    # print(f"goal: {goal}")
    # print(f"q_solution: {q_solution}")
# print(result.solution_candidates[mask])