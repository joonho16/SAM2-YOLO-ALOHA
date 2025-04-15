python3 ./imitate_episodes.py \
--task_name grasp_cable \
--ckpt_dir ./ckpt/grasp_cable_new_aug_c30_d100_ft \
--policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
--num_epochs 10  --lr 1e-5 --data_folders original \
--load_model ./ckpt/grasp_cable_new_aug_c30_d100 \
--seed 0

# python3 ./imitate_episodes.py \
# --task_name pick_tomato \
# --ckpt_dir ./ckpt/pick_tomato_new \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 2000  --lr 1e-5 --data_folders yolo aug \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_aug_2_c15 \
# --policy_class ACT --kl_weight 10 --chunk_size 15 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original aug \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_aug_3_hd1024 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 1024 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original aug \
# --seed 0


# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_allstar_c25 \
# --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original aug hgdagger_aug_2 \
# --seed 0


# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_allstar_hd1024 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 1024 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original aug hgdagger_aug_2 \
# --seed 0


# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_aug_4 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original aug \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_original_2 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_yolo \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders yolo \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_hgdagger \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original hgdagger \
# --seed 0