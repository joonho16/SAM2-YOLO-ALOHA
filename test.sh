python ./imitate_episodes.py \
--task_name pick_tomato \
--ckpt_dir ./ckpt/pick_tomato \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 25 --dim_feedforward 3200 --eval --backbone resnet18 --temporal_agg \
--num_epochs 200 --lr 1e-5 --record_episode \
--seed 0
