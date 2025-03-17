# Heterogeneous Environment-aware Multimodal Recommendation with Modality Alignment

This repo provides the source code & data of our paper HEARec: [Heterogeneous Environment-aware Multimodal Recommendation with Modality Alignment](https://github.com/HubuKG/HEARec) 

## Overview

The structure of our model is available for viewing in the ``./img``.


### 1. Prerequisites

```
Python==3.8.19,
Pytorch==2.3.0,
numpy==1.19.2,
pandas==1.12.3
scipy==1.6.0,
tqdm==4.59.0
```

Install all requirements with ``pip install -r requirements.txt``.


### 2. Download data

The project `HEARec/data` only contains the interaction history of each dataset. As needed, you can download the corresponding multimodal data from these links [Baby]({https://drive.google.com/drive/folders/1Fk21441EO1l7wgOOARh2thu4FjgtKWQp), [Sports](https://drive.google.com/drive/folders/1iJtyDmgeYdZsvO5297dNafyPDya21e8D), and [Clothing](https://drive.google.com/drive/folders/1Suzbyc26BEPPLQJzT_5-EDz_wMf6r7u6).

### 3. Training on a local server using PyCharm.

Run HEARec by ``python main.py`` with the default dataset is Baby. Specific dataset selection can be modified in `main.py`.

### 4. Training on a local server using Git Bash.

Run HEARec by ``train.\`` with the default dataset is Baby. Specific dataset selection can be modified in `train.py`.


### 5. Train on a cloud server.

Run ``!bash train.sh``

### 6. Modify specific parameters.

You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`. 
