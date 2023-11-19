# Prompt_sam_cryoPPP

This is the official repository for Prompt_sam_cryoPPP : Adapting Segment Anything Model (SAM) through Prompt-based Learning for Enhanced Protein Identification in Cryo-EM Micrographs (submitted to [RECOMB 2024](https://recomb.org/recomb2024/)).

## A Quick Overview

![](./image/figure1.png)

## Requirement

Download **`model checkpoint`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)** and put it at ./model_checkpoint/

1. `git clone https://github.com/yangyang-69/Prompt_sam_cryoPPP.git`
2. Enter the Prompt_sam_cryoPPP folder `cd Prompt_sam_cryoPPP-main` and run `conda env create -f environment.yaml`
3. activate the conda environment `conda activate sam`

## Get Started

###  Evaluation of native SAM's efficacy

We use SAM’s automatic mask generator generated masks for a sample protein type (EMPIAR ID: 10028) and calculate the dice scores. Run `./notebooks/test_ori_sam.py`, get the mask and dice for testing the native SAM through the following command line.

```
python ./notebooks/test_ori_sam.py -net sam -exp_name test_original_on_10028 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -data_path ./dataset/10028_all
```

### Finetuning SAM

1. Train:  `python ./notebooks/train_finetune.py`
2. Evaluation: The code can automatically evaluate the model on the test set during training.
3. Test and Result Visualization: `python ./notebooks/test_finetune.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

### Head-Prompt SAM

1. Train:  

   ```
   python ./notebooks/train_head.py --data_path ./dataset/10028_split/5 -data_name 10028 -exp_name few_shot_5 -save_path ./model_checkpoint/head
   ```

2. Test and Result Visualization:

   ```
   python ./notebooks/test_head.py --data_path ./dataset/10028_split -data_name 10028 -exp_name few_shot_5 -ckpt ./model_checkpoint/head/10028_5.pth
   ```

   You can use the 'vis_image' function to visualize the segmentation results of the test dataset.


### Prefix-Prompt SAM

1. Train: 

   ```
   python ./notebooks/train_prefix.py -net PromptVit -mod sam_token_prompt -exp_name train_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -b 1 -dataset CryoPPP -data_path ./dataset/10028_split/5 -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

3. Test and Result Visualization: 

   ```
    python ./notebooks/test_prefix.py -net PromptVit -mod sam_token_prompt -exp_name test_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./model_checkpoint/prefix/10028_5.pth -b 1 -dataset CryoPPP -data_path ./dataset/10028_split -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

   You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

### Encoder-Prompt SAM

1. Train:  `python ./notebooks/train_encoder.py`
3. Test and Result Visualization: `python ./notebooks/test_encoder.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

## References
* Dhakal, Ashwin, et al. "A large expert-curated cryo-EM image dataset for machine learning protein particle picking." Scientific Data 10.1 (2023): 392.
* Wu, Junde, et al. "Medical sam adapter: Adapting segment anything model for medical image segmentation." arXiv preprint arXiv:2304.12620 (2023).
* Kirillov, Alexander, et al. "Segment anything." arXiv preprint arXiv:2304.02643 (2023).
* Jia, Menglin, et al. "Visual prompt tuning." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

