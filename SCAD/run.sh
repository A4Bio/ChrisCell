ex_name=scf

# Sorafenib
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 512 -z_dim 128 -ep 80 -la1 5 -mbS 8 -mbT 8 -emb 0

#CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d Sorafenib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0.2 -mbS 32 -mbT 32 -emb 1


# # NVP-TAE684
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 128 -ep 80 -la1 2 -mbS 8 -mbT 8 -emb 0

#CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0 -mbS 32 -mbT 32 -emb 1


# # PLX4720 (451Lu)
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py -e FX -d PLX4720_451Lu -g _norm -s 42 -h_dim 512 -z_dim 256 -ep 80 -la1 1 -mbS 8 -mbT 8 -emb 0

#CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d PLX4720_451Lu -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0 -mbS 32 -mbT 32 -emb 1


# # Etoposide
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Etoposide -g _norm -s 42 -h_dim 512 -z_dim 128 -ep 80 -la1 2 -mbS 8 -mbT 8 -emb 0

#CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d Etoposide -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0.6 -mbS 32 -mbT 32 -emb 1