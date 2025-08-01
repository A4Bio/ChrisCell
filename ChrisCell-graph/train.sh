wandb login --relogin --host=http://10.28.0.24:30109 local-ad2197960b091e396fdd4d99bf26f619c33095f0
cd /guoxiaopeng/wangjue/VQCell/VQCell


CUDA_VISIBLE_DEVICES=0 python ./train/main.py \
		--ex_name  debug \
       		--batch_size 1024 \
		--offline 0 \
       		--lr 5e-4 \
       		--epoch 500 \
		--lr_scheduler 'onecycle' \
		--vq_space 12 \
		--levels 12 \
		--layers 1 \
		--hidden_dim 128 \
		--latent_dim 32 \
		--dataset Cell_dataset_text \
		--topk_gene 200 \
		--data_path '' \




