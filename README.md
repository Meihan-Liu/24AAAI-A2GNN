# Rethinking Propagation for Unsupervised Graph Domain Adaptation
This is the source code of AAAI-2024 paper "[Rethinking Propagation for Unsupervised Graph Domain Adaptation]()" (A2GNN).

![image](https://github.com/Meihan-Liu/24AAAI-A2GNN/blob/main/fig/figure.png)

# Requirements
This code requires the following:
* torch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.1.0
* numpy==1.19.2
* scikit-learn==0.24.2

# Dataset
Datasets used in the paper are all publicly available datasets. You can find [Twitch](https://github.com/benedekrozemberczki/datasets#twitch-social-networks) and [Citation](https://github.com/yuntaodu/ASN/tree/main/data) via the links.

# Cite
If you compare with, build on, or use aspects of A2GNN framework, please consider citing the following paper:

```
@article{liu2024rethinking,
  title={Rethinking Propagation for Unsupervised Graph Domain Adaptation},
  author={Liu, Meihan and Fang, Zeyu and Zhang, Zhen and Gu, Ming and Zhou, Sheng and Wang, Xin and Bu, Jiajun},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024},
  pages={13963-13971}
}
```
