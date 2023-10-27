# LSeg Per-Pixel Feature Extraction

The repo contains the modified implementation of ICLR'22 paper [LSeg](https://arxiv.org/abs/2201.03546), where you can extract per-pixel features for any images. Also, we provide how multi-view LSeg feature fusion is done in the CVPR'23 paper [OpenScene](https://pengsongyou.github.io/openscene). 

## Installation
Follow the official [installation instruction](https://github.com/isl-org/lang-seg#installation) to install the environment.

**Fail in installing LSeg?** You are not alone. To me it is always not easy to install LSeg by following their instruction. I provide how I successfully installed it (under GCC=0.9.3, CUDA=11.3) below.

```bash

conda create -n lseg python=3.8
conda activate lseg

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.4.9
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/@331ecdd5306104614cb414b16fbcd9d1a8d40e1e  # this step takes >5 minutes

pip install git+https://github.com/openai/CLIP.git
pip install timm==0.5.4
pip install torchmetrics==0.6.0
pip install setuptools==59.5.0
pip install imageio matplotlib pandas six
```

Next, download the [official checkpoint](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view) and save in `checkpoints/demo_e200.ckpt`.

**Note**: You should also follow [here](https://github.com/isl-org/lang-seg#data-preparation). So, there should be a `../datasets/` folder on the parent level, and have the corresponding ADE20K data there, even though we don't need really it.


## Extract the Per-Pixel LSeg Feature for Images
If you want to extract LSeg per-pixel features and save locally, please check [lseg_feature_extraction.py](lseg_feature_extraction.py).

```bash
python lseg_feature_extraction.py --data_dir data/example/ --output_dir data/example_output/ --img_long_side 320
```
where 
- `data_dir` is the folder where contains RGB images
- `output_dir` is the folder where saves the corresponding LSeg features
- `img_long_side` is the length of the long side of your image. For example, for an image with a resolution of [640, 480], `img_long_side` is 640.




## Multi-View LSeg Feature Fusion
Here we provide the codes for how multi-view fusion mentioned in **Section 3.1** in [OpenScene](https://arxiv.org/abs/2211.15654) is done with LSeg features.
We provide the codes for fusion on different datasets, including ScanNet, Matterport3D, and nucenes: [`fusion_scannet.py`](fusion_scannet.py), [`fusion_matterport.py`](fusion_matterport.py), and [`fusion_nuscenes.py`](fusion_nuscenes.py).
### Prerequisites
Follow the [instruction](https://github.com/pengsongyou/openscene/tree/main/scripts/feature_fusion#prerequisites) to obtain the processed 2D and 3D data of the corresponding dataset.


### Usage
Take [`fusion_scannet.py`](fusion_scannet.py) as an example, to perform multi-view LSeg feature fusion, your can run:
```bash
python fusion.py --data_dir PATH/TO/scannet_processed  --output_dir PATH/TO/OUTPUT_DIR --process_id_range 0,100 --split train
```
where:
- `data_dir`: path to the pre-processed 2D&3D data
- `output_dir`: output directory to save your fused features
- `openseg_model`: path to the OpenSeg model
- `process_id_range`: only process scenes within the range
- `split`: choose from `train`/`val`/`test` data to process

This multi-view fusion corresponds to the [part](https://github.com/pengsongyou/openscene/tree/main/scripts/feature_fusion) in the OpenScene official repo.

## Key Modifications over the Original LSeg
- Support outputting per-pixel 512-dim feature. See [here](additional_utils/models.py#L102) and [here](modules/models/lseg_net.py#L194).
- When extracting per-pixel features, no multi-scale features are considered, only a single scale. See [here](lseg_feature_extraction.py#L75).
- Change the `crop_size` and `base_size` according to the length of the long side of the image. See [here](lseg_feature_extraction.py#L81-L82).

## Citations
If you find this repo useful, please cite both papers:
```bibtex
@inproceedings{Peng2023OpenScene,
  title     = {OpenScene: 3D Scene Understanding with Open Vocabularies},
  author    = {Peng, Songyou and Genova, Kyle and Jiang, Chiyu "Max" and Tagliasacchi, Andrea and Pollefeys, Marc and Funkhouser, Thomas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
```
and
```bibtex
@inproceedings{Li2022LSeg,
title={Language-driven Semantic Segmentation},
author={Boyi Li and Kilian Q Weinberger and Serge Belongie and Vladlen Koltun and Rene Ranftl},
booktitle={International Conference on Learning Representations},
year={2022},
}
```