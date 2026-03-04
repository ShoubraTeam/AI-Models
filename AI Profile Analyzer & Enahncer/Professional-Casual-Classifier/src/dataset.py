import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import src.config as config
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(source_dir):
    root_data = config.ROOT_DATA_DIR
    train_path = config.TRAIN_DIR
    test_path = config.TEST_DIR

    if os.path.exists(train_path) and os.path.exists(test_path):
        if any(os.scandir(train_path)):
            return

    categories = ['professional', 'casual']
    for cat in categories:
        os.makedirs(os.path.join(train_path, cat), exist_ok=True)
        os.makedirs(os.path.join(test_path, cat), exist_ok=True)

    for category in categories:
        cat_source_path = os.path.join(source_dir, category)
        if not os.path.exists(cat_source_path):
            continue
        images = [f for f in os.listdir(cat_source_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(images) == 0:
            continue

        train_imgs, test_imgs = train_test_split(images, test_size=config.VAL_SPLIT, random_state=42)

        def move_files(files, target_folder):
            dest = os.path.join(root_data, target_folder, category)
            for f in files:
                shutil.copy(os.path.join(cat_source_path, f), os.path.join(dest, f))

        move_files(train_imgs, 'train')
        move_files(test_imgs, 'test')

def get_transforms(cfg, kind="train"):
    if cfg["use_pretrained"]:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    base_transforms = [
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if kind == "train":
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
        return transforms.Compose(aug_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)

def get_loaders(cfg):
    train_transform = get_transforms(cfg, kind="train")
    val_transform = get_transforms(cfg, kind="val")
    workers = 0 if os.name == 'nt' else 2
    if not os.path.exists(config.TRAIN_DIR):
        full_data = datasets.ImageFolder(root=config.ROOT_DATA_DIR)
        val_size = int(len(full_data) * config.VAL_SPLIT)
        indices = list(range(len(full_data)))
        np.random.shuffle(indices)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        train_ds = Subset(datasets.ImageFolder(config.ROOT_DATA_DIR, transform=train_transform), train_indices)
        val_ds = Subset(datasets.ImageFolder(config.ROOT_DATA_DIR, transform=val_transform), val_indices)
        print(f"Auto-Split: {len(train_ds)} Train (with Augmentations) | {len(val_ds)} Val (Clean)")

    else:
        print("Loading from separate folders...")
        train_ds = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
        val_ds = datasets.ImageFolder(root=config.TEST_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=workers)

    return train_loader, val_loader