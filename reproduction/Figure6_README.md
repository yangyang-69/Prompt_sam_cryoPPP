#  Comparison of finetuning and prompting approaches across all protein types 

For each protein type sourced from CryoPPP dataset we randomly selected 10 samples as the training set, with the remaining samples designated as the test set for SAM adaptation. Figure 6 showcases the average Dice scores, accompanied by error bars, derived from both the fine-tuning and the three prompt-based learning techniques. 

<img src="../image/figure6.png" style="zoom: 67%;" />

**Figure 6** Average Dice scores of SAM with head prompt, prefix prompt, encoder prompt, and fine-tuning across all protein types

## Guideline
For reproduction, please download the sample dataset and the corresponding checkpoints, and modify their paths in the given command line example.
By running the sample command line, you can get the **IOU and dice** of each test image and the **average of IOU and dice** of all images.
You can visualize the segmentation results of the test dataset through the **'vis_image'** function.


- **Command Line Arguments**
(放入参数解释)

- **Output presentation**
  ```
      => resuming from xxx.pth
      => loaded checkpoint xxx.pth (epoch x)
      [Your parameter settings]
      test data length: xx xx
      Validation round:   x%|      | 0/50   ['figure name']  iou: xxx  dice: xxx
      2023-11-08 10:48:56,154 - Total score: xxx, IOU: xxx, DICE: xxx || @ epoch x.
  ```

- **dataset:  1024**
  - Access 1 : Baidu Netdisk https:xxx
  - Access 2 : Google Drive  https:xxx
  
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

- checkpoint:
    - Access 1 : Baidu Netdisk https:xxx
    - Access 2 : Google Drive  https:xxx
   
       ```
       📦checkpoint
        ┣ 📂Figure6
        ┃ ┣ 📂head
        ┃ ┃ ┗ 📜finetune_10028_5.pth
        ┃ ┃ ┗ 📜finetune_10059_5.pth
        ┃ ┃ ┗ 📜finetune_10947_5.pth
       ```

- Command Line: 
   ```
    python ./notebooks/test_finetune.py -net sam -mod sam_fine -exp_name test_finetune_10028 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./checkpoint/FIgure6/finetune/finetune_10028.pth -b 1 -dataset CryoPPP -data_path ./dataset/1024/10028 -image_encoder_configuration 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   ```


## Head-Prompt SAM

- checkpoint: 
     
    - Access 1 : Baidu Netdisk https://pan.baidu.com/s/1FeGriwB9GkgKqgzA9_EJNQ
    - Access 2 : Google Drive  https:xxx

    ```
    📦checkpoint
     ┣ 📂Figure6
     ┃ ┣ 📂head
     ┃ ┃ ┗ 📜head_prompt_10028_5.pth
     ┃ ┃ ┗ 📜head_prompt_10059_5.pth
     ┃ ┃ ┗ 📜head_prompt_10947_5.pth
    ```
       
- Command Line:
   ```
   python ./notebooks/test_head.py -data_path ./dataset/1024/10028 -data_name 10028 -exp_name test_head_10028 -ckpt ./checkpoint/Fgiure6/head/head_prompt_10028_5.pth
   ```

## Prefix-Prompt SAM

- checkpoint:
     
    - Access 1 : Baidu Netdisk https://pan.baidu.com/s/1kL62juXbRveVNhC-uvIejQ
    - Access 2 : Google Drive  https:xxx
         ```
         📦checkpoint
          ┣ 📂Figure6
          ┃ ┣ 📂prefix
          ┃ ┃ ┗ 📜prefix_10028_5.pth
          ┃ ┃ ┗ 📜prefix_10059_5.pth
          ┃ ┃ ┗ 📜prefix_10947_5.pth
         ```

- Command Line:
   ```
    python ./notebooks/test_prefix.py -net PromptVit -mod sam_token_prompt -exp_name test_prefix_all64_token_10028 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./checkpoint/FIgure6/prefix_10028_5.pth -b 1 -dataset CryoPPP -data_path ./dataset/1024/10028 -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

## Encoder-Prompt SAM
- checkpoint:
    - Access 1 : Baidu Netdisk https:xxx
    - Access 2 : Google Drive  https:xxx
   
       ```
       📦checkpoint
        ┣ 📂Figure6
        ┃ ┣ 📂head
        ┃ ┃ ┗ 📜encoder_10028_5.pth
        ┃ ┃ ┗ 📜encoder_10059_5.pth
        ┃ ┃ ┗ 📜encoder_10947_5.pth
       ```

- Command Line: 
   ```
    python ./notebooks/test_encoder.py -net sam -mod sam_adpt -exp_name test_encoder_10028 -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./checkpoint/FIgure6/encoder/encoder_10028.pth -b 1 -dataset CryoPPP -data_path ./dataset/1024/10028 -image_encoder_configuration 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 