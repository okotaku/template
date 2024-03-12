import torch
from PIL import Image
from torchvision import transforms


class DINOScore:
    """DINO Score.

    Args:
    ----
        imgs (list[Image.Image]): The list of ground truth images to use.
        model_name (str): The name of the model to use.
            Defaults to "facebookresearch/dino:main".
        pretrained (str): The pretrained model to use.
            Defaults to "dino_vits16".
        device (str): The device to use.
            Defaults to "cuda:0".

    """

    def __init__(self,
                 imgs: list[Image.Image],
                 model_name: str = "facebookresearch/dino:main",
                 pretrained: str = "dino_vits16",
                 device: str = "cuda:0") -> None:
        self.model = torch.hub.load(model_name, pretrained)
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        imgs_all = [
            self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs]
        imgs_all = torch.cat(imgs_all, dim=0)
        self.image_features = self.model(imgs_all)
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)

        del imgs_all
        torch.cuda.empty_cache()

    def __call__(self,
                 pred: Image.Image) -> float:
        """Calculate the CLIP text score.

        Args:
        ----
            pred (str): The predicted image to use.

        """
        pred = self.preprocess(pred).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_features = self.model(pred)
            pred_features /= pred_features.norm(dim=-1, keepdim=True)
            return torch.mm(
                self.image_features, pred_features.permute(1, 0)).max().item()
