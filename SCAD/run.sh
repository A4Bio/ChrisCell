
# sc: Sorafannib 1024 128 la1 5 32 32 60ep
# zinb: Etoposide 512 256 la1 0.2 32 32 5ep
# sc: PLX 1024 256 la1 0.2 32 32 30ep
# zinb: NVP 1024 128 la1 0.2 32 32 10ep

CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name vqcell_sc -e FX -d Sorafenib -g _norm -s 42 -h_dim 1024 -z_dim 128 -ep 60 -la1 5 -mbS 32 -mbT 32 -emb 1
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name vqcell_zinb -e FX -d Etoposide -g _norm -s 42 -h_dim 512 -z_dim 256 -ep 5 -la1 0.2 -mbS 32 -mbT 32 -emb 1 
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name vqcell_sc -e FX -d PLX4720_451Lu -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 30 -la1 0.2 -mbS 32 -mbT 32 -emb 1
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name vqcell_zinb -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 128 -ep 10 -la1 0.2 -mbS 32 -mbT 32 -emb 1


