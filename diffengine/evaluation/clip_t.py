import open_clip
import torch
from PIL import Image


class CLIPT:
    """CLIP Text Score.

    Args:
    ----
        model_name (str): The name of the model to use.
            Defaults to "ViT-B-32".
        pretrained (str): The pretrained model to use.
            Defaults to "openai".
        device (str): The device to use.
            Defaults to "cuda:0".

    """

    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "openai",
                 device: str = "cuda:0") -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model.eval()
        self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    def __call__(self,
                 img: Image.Image,
                 prompt: str) -> float:
        """Calculate the CLIP text score.

        Args:
        ----
            img (Image.Image): The image to use.
            prompt (str): The prompt to use.

        """
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        text = self.tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return (image_features @ text_features.T)[0, 0].item()
