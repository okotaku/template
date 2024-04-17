# LaVi-Bridge

[Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation](https://arxiv.org/abs/2403.07860)

## Abstract

Text-to-image generation has made significant advancements with the introduction of text-to-image diffusion models. These models typically consist of a language model that interprets user prompts and a vision model that generates corresponding images. As language and vision models continue to progress in their respective domains, there is a great potential in exploring the replacement of components in text-to-image diffusion models with more advanced counterparts. A broader research objective would therefore be to investigate the integration of any two unrelated language and generative vision models for text-to-image generation. In this paper, we explore this objective and propose LaVi-Bridge, a pipeline that enables the integration of diverse pre-trained language models and generative vision models for text-to-image generation. By leveraging LoRA and adapters, LaVi-Bridge offers a flexible and plug-and-play approach without requiring modifications to the original weights of the language and vision models. Our pipeline is compatible with various language models and generative vision models, accommodating different structures. Within this framework, we demonstrate that incorporating superior modules, such as more advanced language models or generative vision models, results in notable improvements in capabilities like text alignment or image quality. Extensive evaluations have been conducted to verify the effectiveness of LaVi-Bridge.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/eec74b38-ffc8-4d63-882c-fd9043633baa"/>
</div>

## Citation

```
@article{zhao2024bridging,
  title={Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation},
  author={Zhao, Shihao and Hao, Shaozhe and Zi, Bojia and Xu, Huaizhe and Wong, Kwan-Yee K},
  journal={arXiv preprint arXiv:2403.07860},
  year={2024}
}
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine wds_train lavi_bridge_v15_laion
```
