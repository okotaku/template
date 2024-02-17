# Debias the Training of Diffusion Models

[Debias the Training of Diffusion Models](https://arxiv.org/abs/2310.08442)

## Abstract

Diffusion models have demonstrated compelling generation quality by optimizing the variational lower bound through a simple denoising score matching loss. In this paper, we provide theoretical evidence that the prevailing practice of using a constant loss weight strategy in diffusion models leads to biased estimation during the training phase. Simply optimizing the denoising network to predict Gaussian noise with constant weighting may hinder precise estimations of original images. To address the issue, we propose an elegant and effective weighting strategy grounded in the theoretically unbiased principle. Moreover, we conduct a comprehensive and systematic exploration to dissect the inherent bias problem deriving from constant weighting loss from the perspectives of its existence, impact and reasons. These analyses are expected to advance our understanding and demystify the inner workings of diffusion models. Through empirical evaluation, we demonstrate that our proposed debiased estimation method significantly enhances sample quality without the reliance on complex techniques, and exhibits improved efficiency compared to the baseline method both in training and sampling processes.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/79b19ec5-d612-44b8-88d2-7d8677b80af5"/>
</div>

## Citation

```
```
