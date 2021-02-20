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
Note the following:
- All data IS pretokenized 
- The input `"example"` field is preprocessed into the phrase-context pair format to be fed into the encoder.

## Run training and evaluation
```python 
python train.py --set data=DATASET_NAME arg1=ARG1 arg2=ARG2# etc...... check out `config/config.yaml` for all arguments
```
Default arguments can be seen and modified in `config/config.yaml`
## Citation
If you find our code, or our work useful - please cite as:
```bibtex
@inproceedings{reid-etal-2020-vcdm,
    title = "{VCDM}: {L}everaging {V}ariational Bi-encoding and {D}eep Contextualized {W}ord {R}epresentations for {I}mproved {D}efinition {M}odeling",
    author = "Reid, Machel  and
      Marrese-Taylor, Edison  and
      Matsuo, Yutaka",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.513",
    doi = "10.18653/v1/2020.emnlp-main.513",
    pages = "6331--6344",
    abstract = "In this paper, we tackle the task of definition modeling, where the goal is to learn to generate definitions of words and phrases. Existing approaches for this task are discriminative, combining distributional and lexical semantics in an implicit rather than direct way. To tackle this issue we propose a generative model for the task, introducing a continuous latent variable to explicitly model the underlying relationship between a phrase used within a context and its definition. We rely on variational inference for estimation and leverage contextualized word embeddings for improved performance. Our approach is evaluated on four existing challenging benchmarks with the addition of two new datasets, {``}Cambridge{''} and the first non-English corpus {``}Robert{''}, which we release to complement our empirical study. Our Variational Contextual Definition Modeler (VCDM) achieves state-of-the-art performance in terms of automatic and human evaluation metrics, demonstrating the effectiveness of our approach.",
}
```
## Contact
If you want to contact us about anything related to the work, feel free to reach out to me at machelreid -at- weblab -dot- t -dot- u-tokyo -dot- ac -dot- jp
## Todo
(21/10/2020)
- Add easier evaluation functionality (add an `--evaluate` argument or something similar)
- Have a better README (hopefully!!)
