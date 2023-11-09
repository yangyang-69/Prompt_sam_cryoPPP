#  Comparisons with existing tools

we compared our techniques with two accessible cryo-EM protein identification tools: crYOLO and 
Topaz.

<img src="./image/figure7.png" alt="image" style="zoom: 25%;" />

**Figure 7**. Figure 7. Visualization of protein particle identification on three sample micrographs achieved by SAM with (a) Head Prompt, (b) Prefix Prompt, (c) Encoder Prompt, (d) Finetuning, along with (e) CrYOLO, and (f) Topaz.

<img src="./image/figure8.png" style="zoom:60%;" />

- **dataset**:  large
    
    ```
    📦large
     ┣ 📂train
     ┃ ┣ 📂images
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜...
     ┃ ┣ 📂labels
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜...
     ┣ 📂valid
     ┃ ┣ 📂images
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜...
     ┃ ┣ 📂labels
     ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┗ 📜...
     ┣ 📂test_7types
     ┃ ┣ 📂10017
     ┃ ┃ ┣ 📂images
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ ...
     ┃ ┃ ┣ 📂labels
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ ...
     ┃ ┣ 📂...
    ```

## Finetuning SAM



## Head-Prompt SAM

1. Train:  `python ./notebooks/train_head.py` and modify the dataset directory in the script by yourself

2. Evaluation: The code can automatically evaluate the model on the test set during training.

3. Test:
   - checkpoint: 
       ```
       📦checkpoint
        ┗ 📜head_prompt_train_head_80_protein_data_add.pth
       ```
   - Result Visualization: `python ./notebooks/test_head.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

## Prefix-Prompt SAM

1. Train:  
   ```
   python ./notebooks/train_prefix.py -net PromptVit -mod sam_token_prompt -exp_name train_prefix_all64_token_large -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -b 1 -dataset CryoPPP -data_path ./dataset/large -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

2. Evaluation: The code can automatically evaluate the model on the test set during training.

3. Test:
   ```
    python ./notebooks/test_prefix.py -net PromptVit -mod sam_token_prompt -exp_name test_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./Checkpoint/Figure7/prefix_large_train_80.pth -b 1 -dataset CryoPPP -data_path ./dataset/large -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```
   - checkpoint: 
       ```
       📦checkpoint
        ┗ 📜prefix_large_train_80.pth
       ```
   - Result Visualization: You can use the 'vis_image' function to visualize the segmentation results of the test dataset.


## Encoder-Prompt SAM
