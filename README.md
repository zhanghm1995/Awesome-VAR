# Awesome VAR Resources
A curated list of papers and resources focused on **Visual AutoRegressive Modeling**, makes GPT-style AR models surpass diffusion transformers in image generation. If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.

📢: The links to the paper's PDF are provided with the `arXiv` badge. You can also easily obtain the approximate release date of this paper from the badge, and we sort these paper in chronological order. If provided, the links to the project pages are provided with the `Project Page` badge, and the `Github Page` badge indicates the availability of the code.

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

### Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis
**Authors**: Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu
<details span>
<summary><b>Abstract</b></summary>
We present Infinity, a Bitwise Visual AutoRegressive Modeling capable of generating high-resolution, photorealistic images following language instruction. Infinity redefines visual autoregressive model under a bitwise token prediction framework with an infinite-vocabulary tokenizer & classifier and bitwise self-correction mechanism, remarkably improving the generation capacity and details. By theoretically scaling the tokenizer vocabulary size to infinity and concurrently scaling the transformer size, our method significantly unleashes powerful scaling capabilities compared to vanilla VAR. Infinity sets a new record for autoregressive text-to-image models, outperforming top-tier diffusion models like SD3-Medium and SDXL. Notably, Infinity surpasses SD3-Medium by improving the GenEval benchmark score from 0.62 to 0.73 and the ImageReward benchmark score from 0.87 to 0.96, achieving a win rate of 66%. Without extra optimization, Infinity generates a high-quality 1024x1024 image in 0.8 seconds, making it 2.6x faster than SD3-Medium and establishing it as the fastest text-to-image model. Models and codes will be released to promote further exploration of Infinity for visual generation and unified tokenizer modeling.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2412.04431-b31b1b.svg)](https://arxiv.org/pdf/2412.04431) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/FoundationVision/Infinity)

### Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient
**Authors**: Zigeng Chen, Xinyin Ma, Gongfan Fang, Xinchao Wang
<details span>
<summary><b>Abstract</b></summary>
In the rapidly advancing field of image generation, Visual Auto-Regressive (VAR) modeling has garnered considerable attention for its innovative next-scale prediction approach. This paradigm offers substantial improvements in efficiency, scalability, and zero-shot generalization. Yet, the inherently coarse-to-fine nature of VAR introduces a prolonged token sequence, leading to prohibitive memory consumption and computational redundancies. To address these bottlenecks, we propose Collaborative Decoding (CoDe), a novel efficient decoding strategy tailored for the VAR framework. CoDe capitalizes on two critical observations: the substantially reduced parameter demands at larger scales and the exclusive generation patterns across different scales. Based on these insights, we partition the multi-scale inference process into a seamless collaboration between a large model and a small model. The large model serves as the 'drafter', specializing in generating low-frequency content at smaller scales, while the smaller model serves as the 'refiner', solely focusing on predicting high-frequency details at larger scales. This collaboration yields remarkable efficiency with minimal impact on quality: CoDe achieves a 1.7x speedup, slashes memory usage by around 50%, and preserves image quality with only a negligible FID increase from 1.95 to 1.98. When drafting steps are further decreased, CoDe can achieve an impressive 2.9x acceleration ratio, reaching 41 images/s at 256x256 resolution on a single NVIDIA 4090 GPU, while preserving a commendable FID of 2.27. 
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2411.17787-b31b1b.svg)](https://arxiv.org/pdf/2411.17787) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/czg1225/code)

### LiteVAR: Compressing Visual Autoregressive Modelling with Efficient Attention and Quantization
**Authors**: Rui Xie, Tianchen Zhao, Zhihang Yuan, Rui Wan, Wenxi Gao, Zhenhua Zhu, Xuefei Ning, Yu Wang
<details span>
<summary><b>Abstract</b></summary>
Visual Autoregressive (VAR) has emerged as a promising approach in image generation, offering competitive potential and performance comparable to diffusion-based models. However, current AR-based visual generation models require substantial computational resources, limiting their applicability on resource-constrained devices. To address this issue, we conducted analysis and identified significant redundancy in three dimensions of the VAR model: (1) the attention map, (2) the attention outputs when using classifier free guidance, and (3) the data precision. Correspondingly, we proposed efficient attention mechanism and low-bit quantization method to enhance the efficiency of VAR models while maintaining performance. With negligible performance lost (less than 0.056 FID increase), we could achieve 85.2% reduction in attention computation, 50% reduction in overall memory and 1.5x latency reduction. To ensure deployment feasibility, we developed efficient training-free compression techniques and analyze the deployment feasibility and efficiency gain of each technique.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2411.17178-b31b1b.svg)](https://arxiv.org/pdf/2411.17178)

### CAR: Controllable Autoregressive Modeling for Visual Generation
**Authors**: Ziyu Yao, Jialin Li, Yifeng Zhou, Yong Liu, Xi Jiang, Chengjie Wang, Feng Zheng, Yuexian Zou, Lei Li
<details span>
<summary><b>Abstract</b></summary>
Controllable generation, which enables fine-grained control over generated outputs, has emerged as a critical focus in visual generative models. Currently, there are two primary technical approaches in visual generation: diffusion models and autoregressive models. Diffusion models, as exemplified by ControlNet and T2I-Adapter, offer advanced control mechanisms, whereas autoregressive models, despite showcasing impressive generative quality and scalability, remain underexplored in terms of controllability and flexibility. In this study, we introduce Controllable AutoRegressive Modeling (CAR), a novel, plug-and-play framework that integrates conditional control into multi-scale latent variable modeling, enabling efficient control generation within a pre-trained visual autoregressive model. CAR progressively refines and captures control representations, which are injected into each autoregressive step of the pre-trained model to guide the generation process. Our approach demonstrates excellent controllability across various types of conditions and delivers higher image quality compared to previous methods. Additionally, CAR achieves robust generalization with significantly fewer training resources compared to those required for pre-training the model. To the best of our knowledge, we are the first to propose a control framework for pre-trained autoregressive visual generation models.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2410.04671-b31b1b.svg)](https://arxiv.org/pdf/2410.04671) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/MiracleDance/CAR)


### STAR: Scale-wise Text-to-image generation via Auto-Regressive representations
**Authors**: Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, Yi Jin
<details span>
<summary><b>Abstract</b></summary>
We present STAR, a text-to-image model that employs scale-wise auto-regressive paradigm. Unlike VAR, which is limited to class-conditioned synthesis within a fixed set of predetermined categories, our STAR enables text-driven open-set generation through three key designs: To boost diversity and generalizability with unseen combinations of objects and concepts, we introduce a pre-trained text encoder to extract representations for textual constraints, which we then use as guidance. To improve the interactions between generated images and fine-grained textual guidance, making results more controllable, additional cross-attention layers are incorporated at each scale. Given the natural structure correlation across different scales, we leverage 2D Rotary Positional Encoding (RoPE) and tweak it into a normalized version. This ensures consistent interpretation of relative positions across token maps at different scales and stabilizes the training process. Extensive experiments demonstrate that STAR surpasses existing benchmarks in terms of fidelity,image text consistency, and aesthetic quality. Our findings emphasize the potential of auto-regressive methods in the field of high-quality image synthesis, offering promising new directions for the T2I field currently dominated by diffusion methods.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2406.10797-b31b1b.svg)](https://arxiv.org/pdf/2406.10797) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/krennic999/STAR)

### ControlVAR: Exploring Controllable Visual Autoregressive Modeling
**Authors**: Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, Bhiksha Raj
<details span>
<summary><b>Abstract</b></summary>
Conditional visual generation has witnessed remarkable progress with the advent of diffusion models (DMs), especially in tasks like control-to-image generation. However, challenges such as expensive computational cost, high inference latency, and difficulties of integration with large language models (LLMs) have necessitated exploring alternatives to DMs. This paper introduces ControlVAR, a novel framework that explores pixel-level controls in visual autoregressive (VAR) modeling for flexible and efficient conditional generation. In contrast to traditional conditional models that learn the conditional distribution, ControlVAR jointly models the distribution of image and pixel-level conditions during training and imposes conditional controls during testing. To enhance the joint modeling, we adopt the next-scale AR prediction paradigm and unify control and image representations. A teacher-forcing guidance strategy is proposed to further facilitate controllable generation with joint modeling. Extensive experiments demonstrate the superior efficacy and flexibility of ControlVAR across various conditional generation tasks against popular conditional DMs, \eg, ControlNet and T2I-Adaptor.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2406.09750-b31b1b.svg)](https://arxiv.org/pdf/2406.09750) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/lxa9867/ControlVAR)


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

### G3PT: Unleash the power of Autoregressive Modeling in 3D Generation via Cross-scale Querying Transformer
**Authors**: Jinzhi Zhang, Feng Xiong, Mu Xu
<details span>
<summary><b>Abstract</b></summary>
Autoregressive transformers have revolutionized generative models in language processing and shown substantial promise in image and video generation. However, these models face significant challenges when extended to 3D generation tasks due to their reliance on next-token prediction to learn token sequences, which is incompatible with the unordered nature of 3D data. Instead of imposing an artificial order on 3D data, in this paper, we introduce G3PT, a scalable coarse-to-fine 3D generative model utilizing a cross-scale querying transformer. The key is to map point-based 3D data into discrete tokens with different levels of detail, naturally establishing a sequential relationship between different levels suitable for autoregressive modeling. Additionally, the cross-scale querying transformer connects tokens globally across different levels of detail without requiring an ordered sequence. Benefiting from this approach, G3PT features a versatile 3D generation pipeline that effortlessly supports diverse conditional structures, enabling the generation of 3D shapes from various types of conditions. Extensive experiments demonstrate that G3PT achieves superior generation quality and generalization ability compared to previous 3D generation methods. Most importantly, for the first time in 3D generation, scaling up G3PT reveals distinct power-law scaling behaviors.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2409.06322-b31b1b.svg)](https://arxiv.org/pdf/2409.06322)

## 3D Perception
### Scalable Autoregressive Monocular Depth Estimation
**Authors**: Jinhong Wang, Jian Liu, Dongqi Tang, Weiqiang Wang, Wentong Li, Danny Chen, Jintai Chen, Jian Wu
<details span>
<summary><b>Abstract</b></summary>
This paper shows that the autoregressive model is an effective and scalable monocular depth estimator. Our idea is simple: We tackle the monocular depth estimation (MDE) task with an autoregressive prediction paradigm, based on two core designs. First, our depth autoregressive model (DAR) treats the depth map of different resolutions as a set of tokens, and conducts the low-to-high resolution autoregressive objective with a patch-wise casual mask. Second, our DAR recursively discretizes the entire depth range into more compact intervals, and attains the coarse-to-fine granularity autoregressive objective in an ordinal-regression manner. By coupling these two autoregressive objectives, our DAR establishes new state-of-the-art (SOTA) on KITTI and NYU Depth v2 by clear margins. Further, our scalable approach allows us to scale the model up to 2.0B and achieve the best RMSE of 1.799 on the KITTI dataset (5% improvement) compared to 1.896 by the current SOTA (Depth Anything). DAR further showcases zero-shot generalization ability on unseen datasets. These results suggest that DAR yields superior performance with an autoregressive prediction paradigm, providing a promising approach to equip modern autoregressive large models (e.g., GPT-4o) with depth estimation capabilities.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2411.11361-b31b1b.svg)](https://arxiv.org/pdf/2411.11361)

### DepthART: Monocular Depth Estimation as Autoregressive Refinement Task
**Authors**: Bulat Gabdullin, Nina Konovalova, Nikolay Patakin, Dmitry Senushkin, Anton Konushin
<details span>
<summary><b>Abstract</b></summary>
Despite recent success in discriminative approaches in monocular depth estimation its quality remains limited by training datasets. Generative approaches mitigate this issue by leveraging strong priors derived from training on internet-scale datasets. Recent studies have demonstrated that large text-to-image diffusion models achieve state-of-the-art results in depth estimation when fine-tuned on small depth datasets. Concurrently, autoregressive generative approaches, such as the Visual AutoRegressive modeling~(VAR), have shown promising results in conditioned image synthesis. Following the visual autoregressive modeling paradigm, we introduce the first autoregressive depth estimation model based on the visual autoregressive transformer. Our primary contribution is DepthART -- a novel training method formulated as Depth Autoregressive Refinement Task. Unlike the original VAR training procedure, which employs static targets, our method utilizes a dynamic target formulation that enables model self-refinement and incorporates multi-modal guidance during training. Specifically, we use model predictions as inputs instead of ground truth token maps during training, framing the objective as residual minimization. Our experiments demonstrate that the proposed training approach significantly outperforms visual autoregressive modeling via next-scale prediction in the depth estimation task. The Visual Autoregressive Transformer trained with our approach on Hypersim achieves superior results on a set of unseen benchmarks compared to other generative and discriminative baselines.
</details>

[![arXiv](https://img.shields.io/badge/arXiv-2409.15010-b31b1b.svg)](https://arxiv.org/pdf/2409.15010)