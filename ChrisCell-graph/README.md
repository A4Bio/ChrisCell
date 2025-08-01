# [ChrisCell-Graph] 

ChrisCell-Graph is a deep learning framework designed to automate the identification of cell states. It utilizes a VQ module to generate a quantized cell state based on single-cell data such as scRNA-seq and scATAC-seq. Additionally, ChrisCell-Graph incorporates a graph attention module to enhance cell representation and improve the interpretability of cell states by highlighting the importance of each gene or property. With these two key components, ChrisCell-Graph provides a robust, effective, efficient, and scalable solution for automating the identification of cell states.


## Table of Contents
- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Running the Example](#running-the-example)
  - [Using Custom Data](#using-custom-data)
- [References](#references)
- [Contact](#contact)
- [License](#license)

## Background

Traditional cell classifications in biology, primarily based on broad somatic types, were largely shaped by the resolution and scope of available measurement techniques. However, this constrained classification has hindered the understanding of cell biology and the development of cell therapies. Recent advances in cell profiling technologies enable the creation of comprehensive cell state models and enlightenment into cell biology and therapy development with significant implications for human health. These technology advances provide the crucial oppotunities to investigate the celluar versitility to a better extent.


## Features

- **Effective Quantization**: ChrisCell-Graph effectively represents high-dimensional and multi-modal single-cell data using a single token, minimizing information loss.
- **Generalizability**: ChrisCell-Graph can be applied to a wide range of single-cell tasks, including cellular heterogeneity analysis, state gene discovery, gene-gene interaction analysis, perturb-seq analysis, and multi-omics data integration.
- **Interprebility**: The graph attention module provides attention weights that help interpret the significance of each gene or property in relation to the cell state.
- **Controllability and Scalability**: ChrisCell-Graph is a scalable framework, allowing users to apply it to any single-cell data set with a user-defined state size tailored to specific requirements.

For more details on the performance and benchmarking, please refer to our paper.


## Quick Start

To quickly try out ChrisCell-Graph using an example dataset, run the following command:

```
bash run_example.sh
```

This script runs the `inference.py` script with sample data provided in the `examples` folder. It uses a sample density map and a ground truth PDB file for evaluation.

We also provide an example tutorial in `quick_start.ipynb`.

## Usage

### Command-line Arguments

The `inference.py` script supports several command-line arguments:

| Argument                 | Description                                             | Default                             |
|--------------------------|---------------------------------------------------------|-------------------------------------|
| `--level`                | The cell state size. level=12: 2**12 = 4096             | None                                |
| `--data_path`            | Path to the dataset.                                    | None                                |
| `--model_path`           | Path to the pretrained model checkpoint.                | `pretrained_model/checkpoint.pt`    |
| `--output_name`          | Name for the umap visualization.                        | `example`                           |
| `--device`               | Device to run the model on (`cpu` or `cuda`).           | `cuda`                              |
| `--verbose`              | Enable verbose output for debugging.                    | Disabled                            |

### Running the Example

You can run the example directly from the command line:

```bash
python inference.py
```

### Using Custom Data

To use ChrisCell-Graph with your own data, you need to provide a scRNA-seq or scATAC-seq dataset. For example:

```bash
python inference.py --data_path /path/to/your/dataset --output_name your_defined_name  --level 12 --device cuda
```

### Training ChrisCell-Graph

To train ChrisCell-Graph with your own data, you can replace the `--data_path` with your dataset in `train.sh` and run the script `train.sh`:

```bash
bash train.sh
```



