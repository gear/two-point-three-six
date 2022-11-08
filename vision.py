import torch
from PIL import Image
from torchvision import transforms


def tensor_to_PIL(image, 
                  org_mean=[0.485, 0.456, 0.406], 
                  org_std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized image back to PIL image. Default is ImageNet values.
    """

    inv_normalize = transforms.Normalize(
        mean=[-i/j for (i,j) in zip(org_mean, org_std)],
        std=[1/j for j in org_std]
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image
