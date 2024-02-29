# Development Environment Options

## Docker

Below are the quick steps for installing and running dreambooth training using Docker:

```bash
git clone https://github.com/okotaku/template
cd sdxlengine
docker compose up -d
docker compose exec template diffengine train stable_diffusion_xl_dreambooth_lora_dog
```

## Devcontainer

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

# Get Started

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
