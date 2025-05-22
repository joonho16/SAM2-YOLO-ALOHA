python3 ./imitate_episodes.py \
--task_name grasp_cable_yaskawa \
--ckpt_dir ./ckpt/grasp_cable_yaskawa \
--policy_class ACT --kl_weight 10 --chunk_size 15 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --eval --backbone resnet18 --temporal_agg \
--num_epochs 200 --lr 1e-5 \
--seed 23

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_please_ts \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --eval --backbone resnet18 --record_episode --temporal_agg \
# --num_epochs 200 --lr 1e-5 --task_space \
# --seed 23

# python3 ./imitate_episodes.py \
# --task_name home_pose \
# --ckpt_dir ./ckpt/test \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --eval --backbone resnet18 --temporal_agg --record_episode \
# --num_epochs 200 --lr 1e-5 --task_space \
# --seed 23

# python ./imitate_episodes.py \
# --task_name pick_tomato \
# --ckpt_dir ./ckpt/pick_tomato2_only_image \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --eval --backbone resnet18 --temporal_agg --record_episode \
# --task_space --vel_control \
# --num_epochs 200 --lr 1e-5 \
# --seed 0