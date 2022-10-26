

## Installation

This repo is tested on Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for examples). 

For the running environment, use the code under transformers to install the transformers package in the docker as below.
```bash
cd transformers
pip install .
```
The docker already has PyTorch installed. 

## XGen code
The code for XGen is saved under the mobilebert directory. You can find the train_script_main.py file and xgen.json under mobilebert_config directory.
When running in XGen, you only need the mobilebert directory. The transformers directory is only used for package installation. Once installed, it is not necessary. 

The dataset is squad v1.1 which is also prepared in the code. The dataset is just two json files (train and test) with texts data. It is OK to modify the data such as delete some data as long as not changing the data format. 
In the train_script_main.py, there are two arguments to specify the location of train and test data. You can change these arguments to use other datasets. 