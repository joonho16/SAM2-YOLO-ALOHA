python3 ./imitate_episodes.py \
--task_name grasp_obj \
--ckpt_dir ./ckpt/grasp_obj \
--policy_class ACT --kl_weight 10 --chunk_size 15 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --backbone resnet18 \
--num_epochs 2500 --lr 1e-5 \
--data_folders hammer -1  aug_hammer/bgr -1 aug_hammer/crop -1 aug_hammer/dark -1 aug_hammer/light -1 nail -1  aug_nail/bgr -1 aug_nail/crop -1 aug_nail/dark -1 aug_nail/light -1 \
--seed 0

