# Prepare Data
We have processed the embeddings of all drug data where you can download the data in https://doi.org/10.6084/m9.figshare.29826536.v1. Then you need to unzip this file and place it in the SCAD/data directory.

# Demo Usage
```
# cd to the SCAD folder
# Sorafenib
## without embedding
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 512 -z_dim 128 -ep 20 -la1 5 -mbS 8 -mbT 8 -emb 0

## with embedding
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0.2 -mbS 32 -mbT 32 -emb 1
```
Follow the command in `run.sh` for more results. It will take several minutes to train the model. You can reproduce our results by 
```
bash run.sh
```



## Expected output

|Drug | Method | Single Cell AUROC |
|-----|--------|-------------------|
|Sorafenib|ChrisCell|0.872|
|NVP-TAE684|ChrisCell|0.84|
|PLX4720_451Lu|ChrisCell|0.772|
|Etoposide|ChrisCell|0.762|

## requirements
```
scanpy
pandas
pytorch
scikit-learn
```
