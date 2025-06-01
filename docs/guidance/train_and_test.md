# Training & Testing

We trained **MAFF-Net** for a total of **60 epochs** on a single **NVIDIA RTX 4090 GPU** with a batch size of **4**.

The final trained checkpoints for both **VoD** and **TJ4DRadSet** are available for download:

- 🔗 [Download VoD Checkpoints](https://github.com/TRV-Lab/MAFF-Net/releases/download/checkpoints/vod2025010810360.pth)
- 🔗 [Download TJ4DRadSet Checkpoints](https://github.com/TRV-Lab/MAFF-Net/releases/download/checkpoints/tj4d2025060814210.pth)

Place all checkpoint files under the following directory: MAFF-Net/checkpoints/.
## Test and evaluate the models
* Test with a model: 
```shell script
python tools/test.py --cfg_file tools/cfgs/MAFF-Net/MAFF-Net_vod.yaml --batch_size 1 --ckpt checkpoints/vod2025010810360.pth
python tools/test.py --cfg_file tools/cfgs/MAFF-Net/MAFF-Net_TJ4D.yaml --batch_size 1 --ckpt checkpoints/tj4d2025060814210.pth
```


## Train a model


* Train with a single GPU:
```shell script
python tools/train.py --cfg_file tools/cfgs/MAFF-Net/MAFF-Net_vod.yaml
python tools/train.py --cfg_file tools/cfgs/MAFF-Net/MAFF-Net_TJ4D.yaml
```
