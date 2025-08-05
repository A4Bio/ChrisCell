The commands in the `DeepCDR/prog/run.sh` can be used to train the CDR prediction model. It will take several minutes. 

## Demo Usage
```
# cd to the DeepCDR folder
mkdir log
mkdir checkpoint
cd ./prog/
## baseline model
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py -use_gexp > ../log/base_model.log 2>&1
## embedding based model
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py --ckpt_name models.ckpt  -use_gexp > ../log/chriscell.log 2>&1

```

## Expected output
For baseline model, the output will be like:`The overall Pearson's correlation is 0.8371.`

For embedding based model, the output will be like:`The overall Pearson's correlation is 0.923.`

## Requirements
```
Keras==2.1.4
TensorFlow==1.13.1
hickle >= 2.1.0
```