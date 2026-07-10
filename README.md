# HiFT

Official implementation of **"HiFT: Hierarchical Frequency-to-Time Recalibration for Efficient Speaker Embedding Learning"**.

This repository provides the model definition files for HiFT.

## Installation

This repository provides the model architecture implementation and reference files for HiFT, including the model definition, a training configuration, VoxCeleb-style data lists/trials, and a reference training/evaluation script.

## Repository Structure

```text
conf/HiFT.yaml                 # reference training configuration
models/HiFT.py                 # HiFT model architecture implementation
data/voxmini/wav.scp           # Kaldi-style training audio list
data/vox1/wav.scp              # Kaldi-style evaluation audio list
data/vox1/trials/              # VoxCeleb1 trial lists
run.sh                         # reference training/evaluation pipeline
requirements.txt               # Python dependencies
LICENSE                        # MIT License
end-to-end figure of HiFT.pdf  # an end-to-end figure of the HiFT-based model
```

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n hift python=3.9
conda activate hift
pip install -r requirements.txt
```

Please install a PyTorch version compatible with your CUDA environment before running training or evaluation.

models.py      # model implementation
## Model Usage

The proposed model can be imported from `models/HiFT.py`:

```python
from models.HiFT import DFResNet155_FullHiFT

model = DFResNet155_FullHiFT(
    feat_dim=80,
    embed_dim=256,
    pooling_func="TSTP",
    two_emb_layer=False,
)
```

## Data Preparation

This repository provides Kaldi-style metadata files and trial lists. The raw VoxCeleb audio files are not included. Please download the required datasets separately and update the paths in `data/voxmini/wav.scp` and `data/vox1/wav.scp` according to your local environment.

## Training and Evaluation

`run.sh` provides a reference WeSpeaker-style pipeline for data preparation, training, model averaging, embedding extraction, scoring, score normalization, calibration, and model export.

Example:

```bash
bash run.sh --stage 3 --stop_stage 6 --config conf/HiFT.yaml --gpus "[0]"
```

The script assumes the corresponding WeSpeaker toolkit structure and helper scripts are available in the working directory.

## Citation

If you find this repository useful, please cite our paper:

```text
HiFT: Hierarchical Frequency-to-Time Recalibration for Efficient Speaker Embedding Learning
```

## License

This project is licensed under the MIT License.