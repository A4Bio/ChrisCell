CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_name norman --batch_size 6 --accumulation_steps 5 --lr 5e-4 --epochs 5 --model_type 'maeautobin' --result_dir ./results/best_norman > results/norman/train6.log & 

