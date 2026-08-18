# ObjectNavigation

This repository contains the reproduction and experimental development of
**AKGVP: Aligning Knowledge Graph with Visual Perception for Object-goal Navigation (ICRA 2024)**.

The current implementation is based on the official AKGVP repository:

https://github.com/nuoxu/AKGVP

Base commit:

`40ed00d`

For reproduction details and experimental modifications, see
[`REPRODUCTION.md`](REPRODUCTION.md).


## Setup
- Clone the repository and move into the top-level directory `cd AKGVP`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate akgvp`
- Our settings of dataset follows previous works, please refer to [HOZ](https://github.com/sx-zhang/HOZ.git) and [L-sTDE](https://github.com/sx-zhang/Layout-based-sTDE.git) for AI2THOR.
- After placing the dataset, use CLIP to generate image features. `python create_image_feat.py`
- For zero-shot navigation, lines 70-73 in `runners/a3c_train.py` can be enabled. In this way, certain categories will be filtered during the training.

## Training and Evaluation

### Train the AKGVP model
```shell
python main.py \
      --title AKGVPModel \
      --model AKGVPModel \
      --workers 4 \
      --gpu-ids 0 \
      --images-file-name clip_featuremap.hdf5