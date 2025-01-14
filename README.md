# [MICCAI'24] Multi-Modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72086-4_48) | [Project Page](https://jaeyoonssim.github.io/publications/miccai2024/miccai2024_v2.github.io-main/index.html)

- This is the official PyTorch implementation of Multi-Modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification.

![overview](img/overview.png)

## Abstract
The graphical representation of the brain offers critical insights into diagnosing and prognosing neurodegenerative disease via relationships between regions of interest (ROIs). Despite recent emergence of various Graph Neural Networks (GNNs) to effectively capture the relational information, there remain inherent limitations in interpreting the brain networks. Specifically, convolutional approaches ineffectively aggregate information from distant neighborhoods, while attention-based methods exhibit deficiencies in capturing node-centric information, particularly in retaining critical characteristics from pivotal nodes. These shortcomings reveal challenges for identifying disease-specific variation from diverse features from different modalities. In this regards, we propose an integrated framework guiding diffusion process at each node by a downstream transformer where both short- and long-range properties of graphs are aggregated via diffusion-kernel and multi-head attention respectively. We demonstrate the superiority of our model by improving performance of pre-clinical Alzheimer’s disease (AD) classification with various modalities. Also, our model adeptly identifies key ROIs that are closely associated with the preclinical stages of AD, marking a significant potential for early diagnosis and prevision of the disease.

## Citation
If you find our work useful for your research, please cite the our paper:
```
@inproceedings{sim2024multi,
  title={Multi-modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification},
  author={Sim, Jaeyoon and Lee, Minjae and Wu, Guorong and Kim, Won Hwa},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={511--521},
  year={2024},
  organization={Springer}
}
```
