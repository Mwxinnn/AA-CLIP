# AA-CLIP Reproduction: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP
**Reproducibility fork of [the official AA-CLIP implementation](https://github.com/Mwxinnn/AA-CLIP)**

---

## Overview

This repository is a **fork of [the official AA-CLIP implementation](https://github.com/Mwxinnn/AA-CLIP)**.

The goal of this fork is to:

- Reproduce the experiments reported in the AA-CLIP paper
- Fix minor issues and improve code reliability
- Provide **public experiment logs**

The reproduced experiments serve as a **reference baseline** for comparing:

- the results reported in the original paper
- the results obtained from this reproduction
- future extensions or improvements built on top of AA-CLIP.

This repository therefore focuses primarily on **reproducibility and transparent evaluation**.

---
# Quick Start

## 1. Installation

Clone this reproducibility fork:

```bash
git clone https://github.com/vantienpham/AA-CLIP.git
cd AA-CLIP

conda create -n aaclip python=3.10 -y
conda activate aaclip

pip install -r requirements.txt
```
### 2. Datasets
The datasets can be downloaded from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BrainMRI, LiverCT, Retinafrom](https://drive.google.com/drive/folders/1La5H_3tqWioPmGN04DM1vdl3rbcBez62?usp=sharing) from [BMAD](https://github.com/DorisBao/BMAD), [CVC-ColonDB, CVC-ClinicDB, Kvasir, CVC-300](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) from Polyp Dataset.

Put all the datasets under ``./data`` and use jsonl files in ``./dataset/metadata/``.

To reproduce the results with few-shot training, you can generate corresponding jsonl files and put them in ``./dataset/metadata/{$dataset}`` with ``{$shot}-shot.jsonl`` as the file name. For few-shot training, we use ``$shot`` samples from each category to train the model. Use `generate_fewshot_jsonl.py` to generate corresponding jsonl file, for example:
```bash
python generate_fewshot_jsonl.py --input_jsonl dataset/metadata/VisA/full-shot.jsonl --output_dir dataset/metadata/VisA/ --shot 2
```
### 3. Training & Evaluation
Download the weight of OpenCLIP ViT-L-14-336px and put it under ```./model/```.

Use `train.py` to train the model:
```bash
python train.py --shot $shot --save_path $save_path
```
The training script automatically evaluates the model on the test datasets after training.