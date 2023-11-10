#  Comparison of finetuning and prompting approaches across all protein types 

For each protein type sourced from CryoPPP dataset we randomly selected 10 samples as the training set, with the remaining samples designated as the test set for SAM adaptation. Figure 6 showcases the average Dice scores, accompanied by error bars, derived from both the fine-tuning and the three prompt-based learning techniques. 

<img src="../image/figure6.png" style="zoom: 67%;" />

**Figure 6** Average Dice scores of SAM with head prompt, prefix prompt, encoder prompt, and fine-tuning across all protein types

- **dataset**:  1024
  
    ```
    📦1024
     ┣ 📂10028
     ┃ ┣ 📂train
     ┃ ┃ ┣ 📂images
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┣ 📂labels
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ 📜image2.png
     ┃ ┣ 📂valid
     ┃ ┃ ┣ 📂images
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┣ 📂labels
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ 📜image2.png
     ┃ ┣ 📂test
     ┃ ┃ ┣ 📂images
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ 📜image2.png
     ┃ ┃ ┣ 📂labels
     ┃ ┃ ┃ ┗ 📜image1.png
     ┃ ┃ ┃ ┗ 📜image2.png
     ┣ 📂10059
     ┃ ...
     ┣ 📂10947
     ┗ ...
    # Every protein type have the same structure.
    ```

## Finetuning SAM



## Head-Prompt SAM

1. Train:  `python ./notebooks/train_head.py` and modify the dataset directory in the script by yourself

2. Evaluation: The code can automatically evaluate the model on the test set during training.

3. Test:
   - checkpoint: 
     
       - Access 1 : 链接：https://pan.baidu.com/s/1FeGriwB9GkgKqgzA9_EJNQ  提取码：njcz
       - Access 2 : 
       
       ```
       📦checkpoint
        ┣ 📂Figure6
        ┃ ┣ 📂head
        ┃ ┃ ┗ 📜head_prompt_10028_5.pth
        ┃ ┃ ┗ 📜head_prompt_10059_5.pth
        ┃ ┃ ┗ 📜head_prompt_10947_5.pth
       ```
       
   - Result Visualization: `python ./notebooks/test_head.py`.You can use the 'vis_image' function to visualize the segmentation results of the test dataset.

## Prefix-Prompt SAM

1. Train:  
   ```
   python ./notebooks/train_prefix.py -net PromptVit -mod sam_token_prompt -exp_name train_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -b 1 -dataset CryoPPP -data_path ./dataset/10028/5 -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

2. Evaluation: The code can automatically evaluate the model on the test set during training.

3. Test:
   ```
    python ./notebooks/test_prefix.py -net PromptVit -mod sam_token_prompt -exp_name test_prefix_all64_token_10028_5 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./Checkpoint/Figure6/prefix_10028_5.pth -b 1 -dataset CryoPPP -data_path ./dataset/10028 -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```
   - checkpoint: 
   
       - Access 1 : 链接：https://pan.baidu.com/s/1kL62juXbRveVNhC-uvIejQ  提取码：zq92
       - Access 2 : 
   
       ```
       📦checkpoint
        ┣ 📂Figure6
        ┃ ┣ 📂prefix
        ┃ ┃ ┗ 📜prefix_10028_5.pth
        ┃ ┃ ┗ 📜prefix_10059_5.pth
        ┃ ┃ ┗ 📜prefix_10947_5.pth
       ```
   
   - Result Visualization: You can use the 'vis_image' function to visualize the segmentation results of the test dataset.


## Encoder-Prompt SAM
