#  Adaptability introduced by three prompt-based learning approaches to SAM

To assess the adaptability introduced by the three proposed prompt-based learning approaches to SAM, we conducted preliminary tests on three distinct protein types with different training sizes from the CryoPPP dataset (EMPIAR IDs: 10028, 10947, and 10059).

![image](./image/figure4.png)

**Figure 4**. Dice score comparisons of SAM with head prompt, prefix prompt, and encoder prompt on different training sizes using proteins from (a) EMPIAR ID 10028 (b) EMPIAR ID 10947 (c) EMPIAR ID 10059 cryo-EM micrographs.

- **dataset**:  10028, 10947, 10059
    
    ```
    📦10028
     ┣ 📂5
     ┃ ┣ 📂images
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┗ ...
     ┃ ┣ 📂labels
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┗ ...
     ┣ 📂10
     ┃ ...
     ┣ 📂250
     ┃ ┣ 📂images
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┗ ...
     ┃ ┣ 📂labels
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┗ ...
    # 10028, 10947 and 10059 have the same structure
    ```

## Finetuning SAM



## Head-Prompt SAM

1. Train:  `python ./notebooks/train_head.py` and modify the dataset directory in the script by yourself

2. Evaluation: The code can automatically evaluate the model on the test set during training.

3. Test:
   - checkpoint: 
       ```
       📦checkpoint
        ┗ 📜head_prompt_5.pth
        ┗ 📜head_prompt_10.pth
        ┗ 📜head_prompt_20.pth
        ┗ 📜head_prompt_30.pth
        ┗ 📜head_prompt_50.pth
        ┗ 📜head_prompt_100.pth
        ┗ 📜head_prompt_150.pth
        ┗ 📜head_prompt_200.pth
        ┗ 📜head_prompt_250.pth
       ```
   - Result Visualization: `python ./notebooks/test_head.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

## Prefix-Prompt SAM

1. Train:  
   ```
   python ./notebooks/train_prefix.py -net PromptVit -mod sam_token_prompt -exp_name train_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -b 1 -dataset CryoPPP -data_path ./dataset/10028_split/5 -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

2. Evaluation: The code can automatically evaluate the model on the test set during training.

3. Test:
   ```
    python ./notebooks/test_prefix.py -net PromptVit -mod sam_token_prompt -exp_name test_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./model_checkpoint/prefix/10028_5.pth -b 1 -dataset CryoPPP -data_path ./dataset/10028_split -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```
   - checkpoint: 
       ```
       📦checkpoint
        ┗ 📜prefix_10028_5.pth
        ┗ 📜prefix_10028_10.pth
        ┗ 📜prefix_10028_20.pth
        ┗ 📜prefix_10028_30.pth
        ┗ 📜prefix_10028_50.pth
        ┗ 📜prefix_10028_100.pth
        ┗ 📜prefix_10028_150.pth
        ┗ 📜prefix_10028_200.pth
        ┗ 📜prefix_10028_250.pth
       ```
   - Result Visualization: You can use the 'vis_image' function to visualize the segmentation results of the test dataset.


## Encoder-Prompt SAM
