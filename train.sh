python3 ./imitate_episodes.py \
--task_name pick_tomato \
--ckpt_dir ./ckpt/pick_tomato2_only_image \
--policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 --task_space \
--task_space --vel_control \
--num_epochs 1000 --lr 1e-5 --data_folders original -1  \
--seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_please_ts \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 --task_space \
# --num_epochs 1000 --lr 1e-5 --data_folders original -1  \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_v2.5 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 500 --lr 1e-5 --data_folders debug_hc_1 50 --task_space \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_test_ts \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 2000 --lr 1e-5 --data_folders original -1 \
# --task_space \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_1 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 100 --lr 1e-5 \
# --data_folders original 140 dagger 60 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_2 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 300 --lr 1e-5 \
# --data_folders original 140 dagger 60 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_3 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 10 --lr 1e-5 \
# --data_folders original 140 dagger 140 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_4 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 50 --lr 1e-5 \
# --data_folders original 140 dagger 140 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_5 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 300 --lr 1e-6 \
# --data_folders original 140 dagger 60 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_6 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 100 --lr 1e-6 \
# --data_folders original 140 dagger 140 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_7 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 30 --lr 1e-6 \
# --data_folders original 20 dagger 60 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_ft_8 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 50 --lr 1e-6 \
# --data_folders original 60 dagger 60 dagger/aug/dark 30 dagger/aug/light 30 dagger/aug/crop 30 \
# --seed 0


# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable_new_augall_ts_c30_d140_dag14_2 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --load_model ./ckpt/grasp_cable_new_augall_c30_d140 \
# --num_epochs 200 --lr 1e-5 --data_folders original 130 dagger 130 dagger/aug/dark 40 dagger/aug/light 40 dagger/aug/crop 40 dagger/aug/bgr 40 \
# --seed 0


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