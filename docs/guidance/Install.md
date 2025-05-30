# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04)
* Python 3.8
* PyTorch 1.10.1
* CUDA 11.3 or higher


### Conda MAFFNet Installation

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n MAFFNet python=3.8 -y
conda activate MAFFNet
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**c. Install the dependent libraries as follows:
```shell
pip install -r requirements.txt
pip install spconv-cu113
pip install --upgrade packaging setuptools wheel
```

**d. Install this `MAFF-Net` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```