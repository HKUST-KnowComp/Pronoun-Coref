# Pronoun coreference resolution

This is the source code for NAACL-HLT 2019 paper "Incorporating Context and External Knowledge for Pronoun Coreference Resolution".

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
@inproceedings{zhang2019pronoun,
  author    = {Hongming Zhang and
               Yan Song and
               Yangqiu Song},
  title     = {Incorporating Context and External Knowledge for Pronoun Coreference Resolution},
  booktitle = {Proceedings of NAACL-HLT, 2019},
  year      = {2019}
}
```



#Usage

Before repeating our experiment or train your own model, please setup the environment as follows:

1. Download python 3.6 or above and setup the anaconda environment by: conda env create -f environment.yml
2. Download the train, dev, and test data from: [Data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hzhangal_connect_ust_hk/EqilkHZp0B5DkBrsbHIFx1gBCLDP24pdIXRhVAIMweLI8A?e=1o9hjd)
3. Download and process the word embeddings: ./setup_embedding.sh
4. Setup the pretrain ELMo module by: python cache_elmo train.jsonlines dev.jsonlines test.jsonlines
5. Train your model with: python Train.py YourSettingName
6. Evaluate your model with: python Evaluate.py YourSettingName

# Acknowledgment
We built the training framework based on the original [End-to-end Coreference Resolution](https://github.com/kentonl/e2e-coref).

# Others
If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.