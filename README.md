# DL Framework Comparison

This project aims to compare the most common Deep Learning Frameworks (TensorFlow and PyTorch) using the Intel Image Classification Dataset.

## Prerequisites

### 1. Install `uv`

This project uses `uv` as the dependency manager. You can install it following the instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

The most common way to install `uv` is via curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download the Dataset
Download the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle. Extract the downloaded archive and place the contents into a directory named `data` at the root of this project. 

Your project structure should look like this:

```text
dl-framework-comparison/
├── data/
│   ├── seg_train/
│   ├── seg_test/
│   └── seg_pred/
...
```
