## Data
We now provide the processed datasets used in our experiments. You can download all h5ad files from Figshare https://doi.org/10.6084/m9.figshare.28787489.v2 .

Then you need to  download the `go.csv.zip` from https://www.dropbox.com/s/wl0uxiz5z9dbliv/go.csv.zip?dl=0 and unzip it to the `data/DATASETNAME/` folder.  

## Installation
Install `PyG`, and then do `pip install cell-gears`. Actually we do not use the gears PyPI packages but installing it will install all the dependencies.

## Demo Usage
```
# cd to the GEARS folder
bash train_gears.sh
```

And the results will be saved in the `results` folder.  

## Expected output
You will get a folder named `results/DATASETNAME/0.75/xxx` with the following files:
```
config.pkl
model.pt
params.csv
train.log
```
