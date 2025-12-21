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
TO BE UPDATED
```

## Run Evaluation
To run evaluation, go to repo folder, modify eval.py on ds, param.py for directory, device, etc
```bash
python eval.py <shots> -mode
```
to run evaluation on <shots>, mode = AC or AS for anomaly classification or segmentation

## Get Image segmentation result visually
Modify data directory in param.py, modify directory, shot, object name in segmentation.py to specify where to save result, number of shot, object
```bash
python segmentation.py
```
get result in ur specified directory

## Data
Produce promissing results on [MVtecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad)\
Also available for [VisA](https://github.com/amazon-science/spot-diff?tab=readme-ov-file), (with limitations)

## Acknowledgement
Many codes borrowed from [OpenCLIP](https://github.com/mlfoundations/open_clip) and [caoyunkang](https://github.com/caoyunkang/WinClip), your works helped me a lot during the process!
