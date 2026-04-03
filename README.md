# WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
Unofficial reimplementation of: [CVPR 2023] WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html)
## Citation:
```bash
@InProceedings{Jeong_2023_CVPR,
    author    = {Jeong, Jongheon and Zou, Yang and Kim, Taewan and Zhang, Dongqing and Ravichandran, Avinash and Dabeer, Onkar},
    title     = {WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19606-19616}
}
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run Evaluation
To run evaluation, go to repo folder, modify eval.py on ds, param.py for directory, device, etc
```bash
python3 eval.py <shots> -mode
```
to run evaluation on <shots>, mode = AC or AS for anomaly classification or segmentation

## Get Image segmentation result visually
Perform anomaly segmentation on specified image
```bash
python3 segmentation.py
```
get result in ur specified directory

## Data
Produce promissing results on [MVtecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad)\
Also available for [VisA](https://github.com/amazon-science/spot-diff?tab=readme-ov-file), (with limitations)

## Results
### MVTec-AD
| K | Reported |      |        |         |      |          | Reproduced |      |        |         |      |          | 
|:-:|:--------:|:----:|:------:|:-------:|:----:|:--------:|:----------:|:----:|:------:|:-------:|:----:|:--------:|
|   | AUROC    | AUPR | F1_max | p_AUROC | PRO  | p_F1_max | AUROC      | AUPR | F1_max | p_AUROC | PRO  | p_F1_max |
| 0 | 91.8     | 96.5 | 92.9   | 85.1    | 64.6 | 31.7     | 90.5       | 95.7 | 84.2   | 81.3    | 83.8 | 20.6     |
| 1 | 93.1     | 96.5 | 93.7   | 95.2    | 87.1 | 55.9     | 93.0       | 96.5 | 92.9   | 91.0    | 92.0 | 37.0     |
| 2 | 94.4     | 97.0 | 94.4   | 96.0    | 88.4 | 58.4     | 93.8       | 96.9 | 92.9   | 91.3    | 92.3 | 37.5     |
| 4 | 95.2     | 97.3 | 94.7   | 96.2    | 89.0 | 59.5     | 93.7       | 96.9 | 93.7   | 91.3    | 92.4 | 37.4     |
### VisA
| K | Reported |      |        |         |      |          | Reproduced |      |        |         |      |          | 
|:-:|:--------:|:----:|:------:|:-------:|:----:|:--------:|:----------:|:----:|:------:|:-------:|:----:|:--------:|
|   | AUROC    | AUPR | F1_max | p_AUROC | PRO  | p_F1_max | AUROC      | AUPR | F1_max | p_AUROC | PRO  | p_F1_max |
| 0 | 78.1     | 81.2 | 79.0   | 79.6    | 56.8 | 14.8     | 78.9       | 81.7 | 79.7   | 74.9    | 76.3 | 6.6      |
| 1 | 83.8     | 85.1 | 83.1   | 96.4    | 85.1 | 41.3     | 81.6       | 83.0 | 81.7   | 87.6    | 85.8 | 11.2     |
| 2 | 84.6     | 85.8 | 83.0   | 96.8    | 86.2 | 43.5     | 81.7       | 83.0 | 81.8   | 87.7    | 85.9 | 11.2     |
| 4 | 87.3     | 88.8 | 84.2   | 97.2    | 87.6 | 47.0     | 81.9       | 83.4 | 81.8   | 87.8    | 85.6 | 11.5     |
### Visualized Segmentation Results
#### Zero-shot Segmentation
<img src=images/Zero_segmentation.png>

#### One-shot Segmentation Improvements
<img src=images/1-shot-segmentation.png>

## Acknowledgement
Many codes borrowed from [OpenCLIP](https://github.com/mlfoundations/open_clip) and [caoyunkang](https://github.com/caoyunkang/WinClip), your works helped me a lot during the process!
