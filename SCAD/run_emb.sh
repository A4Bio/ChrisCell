
ex_name=vqcell_max

CUDA_VISIBLE_DEVICES=0 python run_embedding_sc.py  --data_path ./data/split_norm/Target_expr_resp_19264.Etoposide.csv --ckpt_name $ex_name
CUDA_VISIBLE_DEVICES=0 python run_embedding_bulk.py --data_path ./data/split_norm/Source_exprs_resp_19264.Etoposide.csv --ckpt_name $ex_name

CUDA_VISIBLE_DEVICES=0 python run_embedding_sc.py  --data_path ./data/split_norm/Target_expr_resp_19264.NVP-TAE684.csv --ckpt_name $ex_name
CUDA_VISIBLE_DEVICES=0 python run_embedding_bulk.py --data_path ./data/split_norm/Source_exprs_resp_19264.NVP-TAE684.csv --ckpt_name $ex_name

CUDA_VISIBLE_DEVICES=0 python run_embedding_sc.py  --data_path ./data/split_norm/Target_expr_resp_19264.Sorafenib.csv --ckpt_name $ex_name
CUDA_VISIBLE_DEVICES=0 python run_embedding_bulk.py --data_path ./data/split_norm/Source_exprs_resp_19264.Sorafenib.csv --ckpt_name $ex_name

CUDA_VISIBLE_DEVICES=0 python run_embedding_sc.py  --data_path ./data/split_norm/Target_expr_resp_19264.PLX4720_451Lu.csv --ckpt_name $ex_name
CUDA_VISIBLE_DEVICES=0 python run_embedding_bulk.py --data_path ./data/split_norm/Source_exprs_resp_19264.PLX4720_451Lu.csv --ckpt_name $ex_name

cd data/split_norm

python split_data_SCAD_5fold_norm.py --drug Sorafenib --emb 1 --ckpt_name $ex_name
python split_data_SCAD_5fold_norm.py --drug Etoposide --emb 1 --ckpt_name $ex_name
python split_data_SCAD_5fold_norm.py --drug PLX4720_451Lu --emb 1 --ckpt_name $ex_name
python split_data_SCAD_5fold_norm.py --drug NVP-TAE684 --emb 1 --ckpt_name $ex_name

cd ../../

# Sorafenib
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d Sorafenib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0.2 -mbS 32 -mbT 32 -emb 1

# # NVP-TAE684
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0 -mbS 8 -mbT 8 -emb 1


# # PLX4720 (451Lu)
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d PLX4720_451Lu -g _norm -s 42 -h_dim 1024 -z_dim 512 -ep 80 -la1 0 -mbS 8 -mbT 8 -emb 1


# # Etoposide
CUDA_VISIBLE_DEVICES=0 python model/SCAD_train_binarized_5folds-pub.py --ckpt_name $ex_name -e FX -d Etoposide -g _norm -s 42 -h_dim 1024 -z_dim 512 -ep 40 -la1 0.6 -mbS 16 -mbT 16 -emb 1