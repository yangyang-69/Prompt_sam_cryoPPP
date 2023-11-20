#  Comparisons with existing tools

we compared our techniques with two accessible cryo-EM protein identification tools: crYOLO and 
Topaz.

<img src="../image/figure7.png" style="zoom: 33%;" />

**Figure 7**. Figure 7. Visualization of protein particle identification on three sample micrographs achieved by SAM with (a) Head Prompt, (b) Prefix Prompt, (c) Encoder Prompt, (d) Finetuning, along with (e) CrYOLO, and (f) Topaz.

<img src="../image/figure8.png" style="zoom: 67%;" />

## Guideline
For reproduction, please download the sample dataset and the corresponding checkpoints, and modify their paths in the given command line example.
By running the sample command line, you can get the **IOU and dice** of each test image and the **average of IOU and dice** of all images.
You can visualize the segmentation results of the test dataset through the **'vis_image'** function.

### Command Line Arguments:
(放入参数解释)

### Output presentation
```
    => resuming from xxx.pth
    => loaded checkpoint xxx.pth (epoch x)
    [Your parameter settings]
    test data length: xx xx
    Validation round:   x%|      | 0/50   ['figure name']  iou: xxx  dice: xxx
    2023-11-08 10:48:56,154 - Total score: xxx, IOU: xxx, DICE: xxx || @ epoch x.
```

### dataset:  large
  - Access 1 : Baidu Netdisk https:xxx
  - Access 2 : Google Drive  https:xxx
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
- checkpoint:
    - Access 1 : Baidu Netdisk https:xxx
    - Access 2 : Google Drive  https:xxx
   
       ```
       📦checkpoint
        ┣ 📂Figure7
        ┃ ┗ 📜finetune_large_train_80.pth
       ```

- Command Line: 
   ```
    python ./notebooks/test_finetune.py -net sam -mod sam_fine -exp_name test_finetune_large -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./checkpoint/FIgure7/finetune_large_train_80.pth -b 1 -dataset CryoPPP -data_path ./dataset/large/test_7types -image_encoder_configuration 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   ```


## Head-Prompt SAM

- checkpoint: 
     
    - Access 1 : Baidu Netdisk https://pan.baidu.com/s/1eyDr_qr0QwKOyIZG64YBdA  
    - Access 2 : Google Drive  
       
    ```
    📦checkpoint
     ┣ 📂Figure7
     ┃ ┗ 📜head_prompt_large_train_80.pth
    ```
  
- Command Line: （修改一下）
   ```
    python ./notebooks/head_prompt.py 
   ```
## Prefix-Prompt SAM

- checkpoint: 
    - Access 1 : Baidu Netdisk https://pan.baidu.com/s/11SZHYOQKQmC4prY2HgIrHw
    - Access 2 : Google Drive
       
      ```
      📦checkpoint
       ┣ 📂Figure7
       ┃ ┗ 📜prefix_large_train_80.pth
      ```

- Command Line: 
   ```
    python ./notebooks/test_prefix.py -net PromptVit -mod sam_token_prompt -exp_name test_prefix_all64_token_large -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./checkpoint/Figure7/prefix_large_train_80.pth -b 1 -dataset CryoPPP -data_path ./dataset/large/test_7types -NUM_TOKENS 64 -deep_token_block_configuration 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   ```

## Encoder-Prompt SAM
- checkpoint:
    - Access 1 : Baidu Netdisk https:xxx
    - Access 2 : Google Drive  https:xxx
   
       ```
       📦checkpoint
        ┣ 📂Figure7
        ┃ ┗ 📜finetune_large_train_80.pth
       ```

- Command Line: 
   ```
    python ./notebooks/test_encoder.py -net sam -mod sam_adpt -exp_name test_encoder_large -sam_ckpt ./model_checkpoint/sam_vit_h_4b8939.pth -weights ./checkpoint/FIgure7/encoder_large_train_80.pth -b 1 -dataset CryoPPP -data_path ./dataset/large/test_7types -image_encoder_configuration 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
   ```