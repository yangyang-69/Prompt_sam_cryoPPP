# Prompt_sam_cryoPPP

This is the official repository for Prompt_sam_cryoPPP : Adapting Segment Anything Model (SAM) through Prompt-based Learning for Enhanced Protein Identification in Cryo-EM Micrographs.

## A Quick Overview

![image-20231103184258159](C:\Users\gao\AppData\Roaming\Typora\typora-user-images\image-20231103184258159.png)

## Requirement

Download **`model checkpoint`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)** and put it at ./model_checkpoint/

1. Create a virtual environment `conda create -n prompt_sam_cryoPPP python==3.8`
2. `git clone https://github.com/yangyang-69/Prompt_sam_cryoPPP.git`
3. Enter the Prompt_sam_cryoPPP folder `cd Prompt_sam_cryoPPP` and run `pip install -e .`

## Get Started

We provide code for training SAM in four different prompt ways as follows:

### Head-Prompt SAM

1. Train:  `python train_head.py`
2. Evaluation: The code can automatically evaluate the model on the test set during training.
3. Test and Result Visualization: `python test_head.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

### Finetune SAM

1. Train:  `python train_finetune.py`
2. Evaluation: The code can automatically evaluate the model on the test set during training.
3. Test and Result Visualization: `python test_finetune.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

### Prefix-Prompt SAM

1. Train:  `python train_prefix.py`
2. Evaluation: The code can automatically evaluate the model on the test set during training.
3. Test and Result Visualization: `python test_prefix.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

### Encoder-Prompt SAM

1. Train:  `python train_encoder.py`
2. Evaluation: The code can automatically evaluate the model on the test set during training.
3. Test and Result Visualization: `python test_encoder.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

## Acknowledgements

We would like to acknowledge the valuable contribution from OpenAI's GPT-3.5 model for aiding in language editing. This work was funded by the National Institutes of Health [R35-GM126985].

## Reference