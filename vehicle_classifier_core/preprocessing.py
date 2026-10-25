"""Shared image preprocessing for training, evaluation, and inference."""

from torchvision import transforms

IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def inference_transform(config=None):
    config = config or {}
    size = tuple(config.get("size", IMAGE_SIZE))
    mean = tuple(config.get("mean", NORMALIZE_MEAN))
    std = tuple(config.get("std", NORMALIZE_STD))
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def training_transform(config=None, preprocessing=None):
    config = config or {}
    preprocessing = preprocessing or {}
    size = tuple(preprocessing.get("size", IMAGE_SIZE))
    mean = tuple(preprocessing.get("mean", NORMALIZE_MEAN))
    std = tuple(preprocessing.get("std", NORMALIZE_STD))
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=config.get("horizontal_flip_probability", 0.5)),
            transforms.RandomAffine(
                degrees=config.get("rotation_degrees", 15),
                translate=tuple(config.get("translation_fraction", (0.1, 0.1))),
                scale=tuple(config.get("scale_range", (0.9, 1.1))),
            ),
            transforms.ColorJitter(
                brightness=config.get("brightness_jitter", 0.3),
                contrast=config.get("contrast_jitter", 0.3),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=config.get("random_erasing_probability", 0.5),
                scale=tuple(config.get("random_erasing_scale", (0.02, 0.2))),
                ratio=tuple(config.get("random_erasing_ratio", (0.3, 3.3))),
                value=0,
            ),
        ]
    )
