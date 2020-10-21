# README
Code for "VCDM: Leveraging Variational Bi-encoding and Deep Contextualized Word Representations for Improved Definition Modeling", to be presented at EMNLP 2020


## Setup & Data
```bash
git clone https://github.com/machelreid/vcdm.git
cd vcdm
wget https://machelreid.github.io/resources/Reid2020VCDM.zip #contains the oxford, urban (slang), and wiki (wikipedia) datasets
unzip Reid2020VCDM.zip
mv Reid2020VCDM/ data/
chmod +x sentence-bleu # for evaluation using `sent-bleu`
```

## Run training and evaluation
```python 
python train.py --set data=DATASET_NAME arg1=ARG1 arg2=ARG2# etc...... check out `config/config.yaml` for all arguments
```
Default arguments can be seen and modified in `config/config.yaml`

## Todo
(21/10/2020)
- Add easier evaluation functionality (add an `--evaluate` argument or something similar)
- Have a better README (hopefully!!)
