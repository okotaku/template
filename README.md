# DiffEngine

[![build](https://github.com/okotaku/template/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/template/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://template.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/okotaku/template.svg)](https://github.com/okotaku/template/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/okotaku/template.svg)](https://github.com/okotaku/template/issues)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

[📘 Documentation](https://template0.readthedocs.io/en/latest/) |
[🤔 Reporting Issues](https://github.com/okotaku/template/issues/new/choose)

## 📄 Table of Contents

- [DiffEngine](#diffengine)
  - [📄 Table of Contents](#-table-of-contents)
  - [📖 Introduction 🔝](#-introduction-)
  - [🛠️ Installation 🔝](#️-installation-)
    - [🐳 Docker](#-docker)
    - [📦 Devcontainer](#-devcontainer)
  - [👨‍🏫 Get Started 🔝](#-get-started-)
  - [📘 Documentation 🔝](#-documentation-)
  - [📊 Model Zoo 🔝](#-model-zoo-)
  - [🙌 Contributing 🔝](#-contributing-)
  - [🎫 License 🔝](#-license-)
  - [🖊️ Citation 🔝](#️-citation-)
  - [🤝 Acknowledgement 🔝](#-acknowledgement-)

## 📖 Introduction [🔝](#-table-of-contents)

DiffEngine is an open-source toolbox designed for training state-of-the-art Diffusion Models. Packed with advanced features including diffusers and MMEngine, DiffEngine empowers both seasoned experts and newcomers in the field to efficiently create and enhance diffusion models. Stay at the forefront of innovation with our cutting-edge platform, accelerating your journey in Diffusion Models training.

1. **Training state-of-the-art Diffusion Models**: Empower your projects with state-of-the-art Diffusion Models. Explore options like Stable Diffusion, DreamBooth, and LoRA.
2. **Unified Config System and Module Designs**: Thanks to MMEngine, our platform boasts a unified configuration system and modular designs. Easily customize hyperparameters, loss functions, and other crucial settings while maintaining a structured and organized project environment.
3. **Inference with diffusers.pipeline**: Seamlessly transition from training to real-world application. Effortlessly deploy your trained Diffusion Models for inference tasks. Enhance your productivity and project timeline.
4. **Optimized training speed**: Our platform is designed to accelerate training speed. We utilize the Apex, Nvidia NGC Container, `torch.compile`. You can achieve high-quality results in less time, accelerating your project timeline and enhancing your productivity.

## 🛠️ Installation [🔝](#-table-of-contents)

#### 🐳 Docker

Below are the quick steps for installing and running dreambooth training using Docker:

```bash
git clone https://github.com/okotaku/template
cd sdxlengine
docker compose up -d
docker compose exec template diffengine train stable_diffusion_xl_dreambooth_lora_dog
```

#### 📦 Devcontainer

You can also utilize the devcontainer to develop the DiffEngine. The devcontainer is a pre-configured development environment that runs in a Docker container. It includes all the necessary tools and dependencies for developing, building, and testing the DiffEngine.

1. Clone repository:

```
git clone https://github.com/okotaku/template
```

2. Open the cloned repository in Visual Studio Code.

3. Click on the "Reopen in Container" button located in the bottom right corner of the window. This action will open the repository within a devcontainer.

4. Run the following command to start training with the selected config:

```bash
diffengine train stable_diffusion_xl_dreambooth_lora_dog
```

## 👨‍🏫 Get Started [🔝](#-table-of-contents)

DiffEngine makes training easy through its pre-defined configs. These configs provide a streamlined way to start your training process. Here's how you can get started using one of the pre-defined configs:

1. **Choose a config**: You can find various pre-defined configs in the [`configs`](diffengine/configs/) directory of the DiffEngine repository. For example, if you wish to train a DreamBooth model using the Stable Diffusion algorithm, you can use the [`configs/dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](diffengine/configs/dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py).

2. **Start Training**: Open a terminal and run the following command to start training with the selected config:

```bash
diffengine train stable_diffusion_v15_dreambooth_lora_dog
```

3. **Monitor Progress and get results**: The training process will begin, and you can track its progress. The outputs of the training will be located in the `work_dirs/stable_diffusion_v15_dreambooth_lora_dog` directory, specifically when using the `stable_diffusion_v15_dreambooth_lora_dog` config.

```
work_dirs/stable_diffusion_v15_dreambooth_lora_dog
├── 20230802_033741
|   ├── 20230802_033741.log  # log file
|   └── vis_data
|         ├── 20230802_033741.json  # log json file
|         ├── config.py  # config file for each experiment
|         └── vis_image  # visualized image from each step
├── step999/unet
|   ├── adapter_config.json  # adapter conrfig file
|   └── adapter_model.bin  # weight for inferencing with diffusers.pipeline
├── iter_1000.pth  # checkpoint from each step
├── last_checkpoint  # last checkpoint, it can be used for resuming
└── stable_diffusion_v15_dreambooth_lora_dog.py  # latest config file
```

An illustrative output example is provided below:

![img](https://github.com/okotaku/template/assets/24734142/e4576779-e05f-42d0-a709-d6481eea87a9)

4. **Inference with diffusers.pipeline**: Once you have trained a model, simply specify the path to the saved model and inference by the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999')
prompt = 'A photo of sks dog in a bucket'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")

image = pipe(
    prompt,
    num_inference_steps=50
).images[0]
image.save('demo.png')
```

## 📘 Documentation [🔝](#-table-of-contents)

For detailed user guides and advanced guides, please refer to our [Documentation](https://template.readthedocs.io/en/latest/):

- [Get Started](https://template.readthedocs.io/en/latest/get_started.html) for get started.

<details>
<summary>Run Guides</summary>

- [Run Stable Diffusion](https://template.readthedocs.io/en/latest/run_guides/run.html)
- [Run DreamBooth](https://template.readthedocs.io/en/latest/run_guides/run_dreambooth.html)
- [Run LoRA](https://template.readthedocs.io/en/latest/run_guides/run_lora.html)
- [Run ControlNet](https://template.readthedocs.io/en/latest/run_guides/run_controlnet.html)
- [Run Inpaint](https://template.readthedocs.io/en/latest/run_guides/run_inpaint.html)

</details>

<details>
<summary>User Guides</summary>

- [Learn About Config](https://template.readthedocs.io/en/latest/user_guides/config.html)
- [Prepare Dataset](https://template.readthedocs.io/en/latest/user_guides/dataset_prepare.html)

</details>

## 📊 Model Zoo [🔝](#-table-of-contents)

<details open>

<div align="center">
  <b>Supported algorithms</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Stable Diffusions</b>
      </td>
      <td>
        <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="diffengine/configs/stable_diffusion/README.md">Stable Diffusion (2022)</a></li>
            <li><a href="diffengine/configs/controlnet/README.md">ControlNet (ICCV'2023)</a></li>
            <li><a href="diffengine/configs/dreambooth/README.md">DreamBooth (CVPR'2023)</a></li>
            <li><a href="diffengine/configs/lora/README.md">LoRA (ICLR'2022)</a></li>
            <li><a href="diffengine/configs/inpaint/README.md">Inpaint</a></li>
      </ul>
      </td>
      <td>
        <ul>
            <li><a href="diffengine/configs/min_snr_loss/README.md">Min-SNR Loss (ICCV'2023)</a></li>
            <li><a href="diffengine/configs/debias_estimation_loss/README.md">DeBias Estimation Loss (2023)</a></li>
            <li><a href="diffengine/configs/offset_noise/README.md">Offset Noise (2023)</a></li>
            <li><a href="diffengine/configs/pyramid_noise/README.md">Pyramid Noise (2023)</a></li>
            <li><a href="diffengine/configs/input_perturbation/README.md">Input Perturbation (2023)</a></li>
            <li><a href="diffengine/configs/timesteps_bias/README.md">Time Steps Bias (2023)</a></li>
            <li><a href="diffengine/configs/v_prediction/README.md">V Prediction (ICLR'2022)</a></li>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>
</details>

## 🙌 Contributing [🔝](#-table-of-contents)

We appreciate all contributions to improve clshub. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmpretrain/blob/main/CONTRIBUTING.md) for the contributing guideline.

## 🎫 License [🔝](#-table-of-contents)

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation [🔝](#-table-of-contents)

If DiffEngine is helpful to your research, please cite it as below.

```
@misc{diffengine2023,
    title = {{DiffEngine}: diffusers training toolbox with mmengine},
    author = {{DiffEngine Contributors}},
    howpublished = {\url{https://github.com/okotaku/diffengine}},
    year = {2023}
}
```

## 🤝 Acknowledgement [🔝](#-table-of-contents)

This repo borrows the architecture design and part of the code from [mmengine](https://github.com/open-mmlab/mmengine) and [diffusers](https://github.com/huggingface/diffusers).

Also, please check the following openmmlab and huggingface projects and the corresponding Documentation.

- [OpenMMLab](https://openmmlab.com/)
- [HuggingFace](https://huggingface.co/)

```
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```

```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

```
@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```
