python3 ./imitate_episodes.py \
--task_name pick_tomato \
--ckpt_dir ./ckpt/pick_tomato_new_yolo \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
--num_epochs 2000  --lr 1e-5 --data_folders yolo \
--seed 0

python3 ./imitate_episodes.py \
--task_name pick_tomato \
--ckpt_dir ./ckpt/pick_tomato_new_original \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
--num_epochs 2000  --lr 1e-5 --data_folders original \
--seed 0

# python3 ./imitate_episodes.py \
# --task_name pick_tomato \
# --ckpt_dir ./ckpt/pick_tomato_new_raw \
# --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 3000  --lr 1e-5 --data_folders original \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name grasp_cable \
# --ckpt_dir ./ckpt/grasp_cable \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 5 --dim_feedforward 3200 --backbone CMA \
# --num_epochs 2000  --lr 1e-5 \
# --seed 0


# python3 ./imitate_episodes.py \
# --task_name cap_a_bottle_with_tactile \
# --ckpt_dir ./ckpt/cap_a_bottle_with_tactile_cma_25_5e6 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone CMA \
# --num_epochs 2000  --lr 5e-6 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name cap_a_bottle_with_tactile \
# --ckpt_dir ./ckpt/cap_a_bottle_with_tactile_cma_30_25_5e6 \
# --policy_class ACT --kl_weight 10 --chunk_size 30 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone CMA \
# --num_epochs 2000  --lr 5e-6 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name cap_a_bottle_with_tactile \
# --ckpt_dir ./ckpt/cap_a_bottle_with_tactile_resnet18_25 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 2000  --lr 5e-6 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name cap_a_bottle \
# --ckpt_dir ./ckpt/cap_a_bottle_resnet18 \
# --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
# --num_epochs 2000  --lr 5e-6 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name cap_a_bottle_with_tactile \
# --ckpt_dir ./ckpt/cap_a_bottle_with_tactile_10_50_1e-5 \
# --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 2 --dim_feedforward 3200 \
# --num_epochs 2000  --lr 1e-5 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name high_five \
# --ckpt_dir ./ckpt/high_five_10_10_1e-5 \
# --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 2 --dim_feedforward 3200 \
# --num_epochs 2000  --lr 1e-5 \
# --seed 0

# python3 ./imitate_episodes.py \
# --task_name put_sponge_in_basket \
# --ckpt_dir ./ckpt/put_sponge_in_basket_20_50_1e-5 \
# --policy_class ACT --kl_weight 20 --chunk_size 50 --hidden_dim 512 --batch_size 2 --dim_feedforward 3200 \
# --num_epochs 2000  --lr 1e-5 \
# --seed 0