# intro-transformer

## Download the toy dataset:

```bash
wget \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -O train_data.text
```

## Install pytorch

```bash
conda create -n intro-transformer python=3.11 pip 
conda activate intro-transformer
pip install torch
```

## Run the training script

```bash
python train.py
```

It will save a `model.pt` file in the current directory.

## Generate text

```bash
python generate.py
```
