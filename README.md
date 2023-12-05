# D<sup>2</sup>ST-Adapter

Official repository for [D<sup>2</sup>ST-Adapter: Disentangled-and-Deformable Spatio-Temporal Adapter for Few-shot Action Recognition](https://arxiv.org/abs/2312.01431). Code will be released soon.

![teaser](figures/Framework.png)

## Abstract

Adapting large pre-trained image models to few-shot action recognition has proven to be an effective and efficient strategy for learning robust feature extractors, which is essential for few-shot learning. Typical fine-tuning based adaptation paradigm is prone to overfitting in the few-shot learning scenarios and offers little modeling flexibility for learning temporal features in video data. In this work we present the Disentangled-and-Deformable Spatio-Temporal Adapter (D<sup>2</sup>ST-Adapter), a novel adapter tuning framework for few-shot action recognition, which is designed in a dual-pathway architecture to encode spatial and temporal features in a disentangled manner. Furthermore, we devise the Deformable Spatio-Temporal Attention module as the core component of D<sup>2</sup>ST-Adapter, which can be tailored to model both spatial and temporal features in corresponding pathways, allowing our D<sup>2</sup>ST-Adapter to encode features in a global view in 3D spatio-temporal space while maintaining a lightweight design. Extensive experiments with instantiations of our method on both pre-trained ResNet and ViT demonstrate the superiority of our method over state-of-the-art methods for few-shot action recognition. Our method is particularly well-suited to challenging scenarios where temporal dynamics are critical for action recognition.

## Citation

```
@misc{pei2023d2stadapter,
      title={D$^2$ST-Adapter: Disentangled-and-Deformable Spatio-Temporal Adapter for Few-shot Action Recognition}, 
      author={Wenjie Pei and Qizhong Tan and Guangming Lu and Jiandong Tian},
      year={2023},
      eprint={2312.01431},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```