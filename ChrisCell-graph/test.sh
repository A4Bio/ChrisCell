

wandb login --relogin --host=http://10.28.0.24:30109 local-ad2197960b091e396fdd4d99bf26f619c33095f0

CUDA_VISIBLE_DEVICES=0 python ./train/test.py \
		--ex_name  test\
       		--batch_size 1000 \
		--lr_scheduler 'onecycle' \
		--vq_space 12 \
		--levels 12 \
		--layers 1 \
		--hidden_dim 128 \
		--latent_dim 32 \
		--data_path  '' \
		--pretrained '/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v4_level8_c2000/checkpoints/model.pt'




