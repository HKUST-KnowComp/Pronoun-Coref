1. Before repeating our experiment or train your own model, please setup the environment as follows:
(1) Download python 3.6 or above and setup the anaconda environment by: conda env create -f environment.yml
(2) Download and process the word embeddings: ./setup_embedding.sh
(3) Setup the pretrain ELMo module by: python cache_elmo train.jsonlines dev.jsonlines test.jsonlines


2. If you simply want to repeat our experiment, you can use the provided traind model by: python Evaluate.py best

3. If you want to train your own model, you should be modify the experiments.conf file to add your own setting and run: python Train.py YourSettingName

PS: please make sure the GPU is available and we defaultly use GPU 0 on your machine.
