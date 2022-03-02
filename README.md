[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/very-deep-transformers-for-neural-machine/machine-translation-on-wmt2014-english-french)](https://paperswithcode.com/sota/machine-translation-on-wmt2014-english-french?p=very-deep-transformers-for-neural-machine)

<h2 align="center">Admin</h2>
<h5 align="center">Understanding the Difficulty of Training Transformers</h5>

Guided by our analyses, we propose **Ad**aptive **M**odel **In**itialization (Admin), which successfully stabilizes previously-diverged Transformer training and achieves better performance, **without introducing additional hyper-parameters**. The design of Admin is half-precision friendly and can be **reparameterized into the original Transformer**. 

In our experiments, Admin [easily stabilize the training of 200L Transformer](https://github.com/LiyuanLucasLiu/Transformer-Clinic/blob/master/nmt-experiments/wmt14_en-de.md). We didn't try deeper Transformer, as [our study](https://arxiv.org/pdf/2008.07772.pdf) shows that balancing depth, width, and encoder-decoder split is better than naïve stacking more layers. Still, we believe Admin can handle the training of Transformer, up to 1000+ layers, if necessary. 

We are in an early-release beta. Expect some adventures and rough edges.

## Table of Contents

- [Introduction](#introduction)
- [Amplification Effect](#dependency-and-amplification-effect)
- [Quick Start](#quick-start-guide)
- [Citation](#citation)

## Introduction
<h5 align="center"><i>What complicates Transformer training?</i></h5>

In our study, we go beyond gradient vanishing and identify an __amplification effect__ that substantially influences Transformer training. 
Specifically, for each layer in a multi-layer Transformer, heavy dependency on its residual branch makes training unstable, yet light dependency leads to sub-optimal performance.

## Dependency and Amplification Effect

Our analysis starts from the observation that Pre-LN is more robust than Post-LN, whereas Post-LN typically leads to a better performance. 
As shown in Figure 1, we find these two variants have different layer dependency patterns. 

<p align="center"><img width="60%" src="img/6_layer_dependency.png"/></p>

With further exploration, we find that for a N-layer residual network, after updating its parameters W to W\*, its outputs change is proportion to the dependency on residual branches. 

<p align="center"><img width="60%" src="img/output_change.png"/></p>

Intuitively, since a larger output change indicates a more unsmooth loss surface, the large dependency complicates training.
Moreover, we propose Admin (**ad**aptive **m**odel **in**itialization), which starts the training from the area with a smoother surface. 
More details can be found in our [paper](https://arxiv.org/abs/2004.08249).

## Quick Start Guide

Our implementation is based on the fairseq package (`python 3.6, torch 1.5/1.6` are recommended). It can be installed by:
```
git clone https://github.com/LiyuanLucasLiu/Transforemr-Clinic.git
cd fairseq
pip install --editable .
```
The guidance for reproducing our results is available at:
- [WMT'14 De-En](nmt-experiments/wmt14_en-de.md)
- [WMT'14 De-Fr](nmt-experiments/wmt14_en-fr.md)
- [IWSLT'14 En-De](nmt-experiments/iwslt14_de-en.md)

Specifically, our implementation requires to first set ```--init-type adaptive-profiling``` and use one GPU for this profiling stage, then set ```--init-type adaptive``` and start training.  

## Citation
Please cite the following papers if you found our model useful. Thanks!

>Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han (2020). Understanding the Difficulty of Training Transformers. Proc. 2020 Conf. on Empirical Methods in Natural Language Processing (EMNLP'20).
```
@inproceedings{liu2020admin,
  title={Understanding the Difficulty of Training Transformers},
  author = {Liu, Liyuan and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu and Han, Jiawei},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)},
  year={2020}
}
```
> Xiaodong Liu, Kevin Duh, Liyuan Liu, and Jianfeng Gao (2020). Very Deep Transformers for Neural Machine Translation. arXiv preprint arXiv:2008.07772 (2020).
```
@inproceedings{liu_deep_2020,
 author = {Liu, Xiaodong and Duh, Kevin and Liu, Liyuan and Gao, Jianfeng},
 booktitle = {arXiv:2008.07772 [cs]},
 title = {Very Deep Transformers for Neural Machine Translation},
 year = {2020}
}
```
