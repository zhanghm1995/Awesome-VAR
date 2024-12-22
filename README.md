# Awesome VAR Resources
A curated list of papers and resources focused on **Visual AutoRegressive Modeling**, makes GPT-style AR models surpass diffusion transformers in image generation. If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.

## Table of contents
- [2D Generation](#2D-Generation)
- [3D Generation](#3D-Generation)
- [3D Perception](#3D-Perception)

<br>

## 2D Generation

### FlowAR: Scale-wise Autoregressive Image Generation Meets Flow Matching
**Authors**: Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, Liang-Chieh Chen
<details span>
<summary><b>Abstract</b></summary>
Autoregressive (AR) modeling has achieved remarkable success in natural language processing by enabling models to generate text with coherence and contextual understanding through next token prediction. Recently, in image generation, VAR proposes scale-wise autoregressive modeling, which extends the next token prediction to the next scale prediction, preserving the 2D structure of images. However, VAR encounters two primary challenges: (1) its complex and rigid scale design limits generalization in next scale prediction, and (2) the generator's dependence on a discrete tokenizer with the same complex scale structure restricts modularity and flexibility in updating the tokenizer. To address these limitations, we introduce FlowAR, a general next scale prediction method featuring a streamlined scale design, where each subsequent scale is simply double the previous one. This eliminates the need for VAR's intricate multi-scale residual tokenizer and enables the use of any off-the-shelf Variational AutoEncoder (VAE). Our simplified design enhances generalization in next scale prediction and facilitates the integration of Flow Matching for high-quality image synthesis. We validate the effectiveness of FlowAR on the challenging ImageNet-256 benchmark, demonstrating superior generation performance compared to previous methods. 
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2412.15205-b31b1b.svg)](https://arxiv.org/pdf/2412.15205) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/OliverRensu/FlowAR)

### [NeurIPS '24] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
**Authors**: Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang
<details span>
<summary><b>Abstract</b></summary>
We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction". This simple, intuitive methodology allows autoregressive (AR) transformers to learn visual distributions fast and generalize well: VAR, for the first time, makes GPT-like AR models surpass diffusion transformers in image generation. On ImageNet 256x256 benchmark, VAR significantly improve AR baseline by improving Frechet inception distance (FID) from 18.65 to 1.73, inception score (IS) from 80.4 to 350.2, with around 20x faster inference speed. It is also empirically verified that VAR outperforms the Diffusion Transformer (DiT) in multiple dimensions including image quality, inference speed, data efficiency, and scalability. Scaling up VAR models exhibits clear power-law scaling laws similar to those observed in LLMs, with linear correlation coefficients near -0.998 as solid evidence. VAR further showcases zero-shot generalization ability in downstream tasks including image in-painting, out-painting, and editing. These results suggest VAR has initially emulated the two important properties of LLMs: Scaling Laws and zero-shot task generalization. We have released all models and codes to promote the exploration of AR/VAR models for visual generation and unified learning.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2404.02905-b31b1b.svg)](https://arxiv.org/pdf/2404.02905) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/FoundationVision/VAR)


## 3D Generation
### SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE
**Authors**: Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, Xingang Pan
<details span>
<summary><b>Abstract</b></summary>
Autoregressive models have demonstrated remarkable success across various fields, from large language models (LLMs) to large multimodal models (LMMs) and 2D content generation, moving closer to artificial general intelligence (AGI). Despite these advances, applying autoregressive approaches to 3D object generation and understanding remains largely unexplored. This paper introduces Scale AutoRegressive 3D (SAR3D), a novel framework that leverages a multi-scale 3D vector-quantized variational autoencoder (VQVAE) to tokenize 3D objects for efficient autoregressive generation and detailed understanding. By predicting the next scale in a multi-scale latent representation instead of the next single token, SAR3D reduces generation time significantly, achieving fast 3D object generation in just 0.82 seconds on an A6000 GPU. Additionally, given the tokens enriched with hierarchical 3D-aware information, we finetune a pretrained LLM on them, enabling multimodal comprehension of 3D content. Our experiments show that SAR3D surpasses current 3D generation methods in both speed and quality and allows LLMs to interpret and caption 3D models comprehensively.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2411.16856-b31b1b.svg)](https://arxiv.org/pdf/2411.16856) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://cyw-3d.github.io/projects/SAR3D/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/cyw-3d/SAR3D)

## 3D Perception
### Scalable Autoregressive Monocular Depth Estimation
**Authors**: Jinhong Wang, Jian Liu, Dongqi Tang, Weiqiang Wang, Wentong Li, Danny Chen, Jintai Chen, Jian Wu
<details span>
<summary><b>Abstract</b></summary>
This paper shows that the autoregressive model is an effective and scalable monocular depth estimator. Our idea is simple: We tackle the monocular depth estimation (MDE) task with an autoregressive prediction paradigm, based on two core designs. First, our depth autoregressive model (DAR) treats the depth map of different resolutions as a set of tokens, and conducts the low-to-high resolution autoregressive objective with a patch-wise casual mask. Second, our DAR recursively discretizes the entire depth range into more compact intervals, and attains the coarse-to-fine granularity autoregressive objective in an ordinal-regression manner. By coupling these two autoregressive objectives, our DAR establishes new state-of-the-art (SOTA) on KITTI and NYU Depth v2 by clear margins. Further, our scalable approach allows us to scale the model up to 2.0B and achieve the best RMSE of 1.799 on the KITTI dataset (5% improvement) compared to 1.896 by the current SOTA (Depth Anything). DAR further showcases zero-shot generalization ability on unseen datasets. These results suggest that DAR yields superior performance with an autoregressive prediction paradigm, providing a promising approach to equip modern autoregressive large models (e.g., GPT-4o) with depth estimation capabilities.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2411.11361-b31b1b.svg)](https://arxiv.org/pdf/2411.11361)