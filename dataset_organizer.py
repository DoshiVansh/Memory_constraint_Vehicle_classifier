# DEPRECATED: This script has been superseded by scripts/ingest_data.py (DVC Stage 0).
# It is kept for historical reference only. Do NOT run it directly — it has no shared
# exclusion list with build_test_data.py and can cause train/test raw-source overlap.
# Use:  dvc repro ingest_data
import os
import shutil
import random
import argparse
from pathlib import Path
from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.reproducibility import seed_everything

# 1. The Professor's Blueprint + The "None" Class
CLASS_MAPPING = {
    "Bus": ["n04487081", "n04146614", "n02924116"],
    "Truck": ["n04467665", "n03417042", "n03345487"],
    "Car": [
        "n02958343", "n03100232", "n03594945", 
        "n03769881", "n04037443", "n04285008"
    ],
    "Bike": ["n03790512", "n02835271"],
    "None": [
        "n06874185", # Traffic light
        "n04328186", # Street sign
        "n03881040", # Park bench
        "n03710193"  # Mailbox
    ]
}

def organize_dataset(source_dir, target_dir, max_images_per_class=2500, seed=1729):
    """
    Extracts specific synsets from a downloaded ImageNet folder, 
    groups them into assignment classes, and balances the dataset.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    rng = random.Random(seed)

    if target_path.exists() and any(target_path.iterdir()):
        raise FileExistsError(
            f"Refusing to merge a newly selected dataset into non-empty {target_path}. "
            "Use a new target path so stale files cannot silently survive a changed image cap."
        )

    # Create target directories
    for index, class_name in enumerate(CLASS_MAPPING):
        os.makedirs(target_path / f"{index}_{class_name}", exist_ok=True)

    print("Starting dataset organization...\n")

    for class_name, synsets in CLASS_MAPPING.items():
        print(f"Processing Class: {class_name}")
        
        # Collect all image paths for the current class across its various synsets
        all_class_images = []
        for synset in synsets:
            synset_folder = source_path / synset
            if synset_folder.exists():
                print(f"  Gathering from synset: {synset}...")
                # The suffix.lower() trick catches .JPG, .jpg, .PNG, .png, and .JPEG
                images = [f for f in synset_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpeg', '.jpg', '.png']]
                all_class_images.extend(images)
            else:
                print(f"  Warning: Synset folder {synset} not found in source.")

        # Shuffle and cap to prevent class imbalance (e.g., Cars having 6x more data than Buses)
        rng.shuffle(all_class_images)
        selected_images = all_class_images[:max_images_per_class]

        # Copy to the new curated dataset folder
        for idx, img_path in enumerate(selected_images):
            # Rename for consistency: e.g., Bus_0001.JPEG
            new_name = f"{class_name}_{idx:04d}.JPEG"
            class_index = list(CLASS_MAPPING).index(class_name)
            dest_path = target_path / f"{class_index}_{class_name}" / new_name
            shutil.copy2(img_path, dest_path)

        print(f"  -> Copied {len(selected_images)} images for {class_name}.\n")

    print(f"Dataset successfully organized at: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    config = load_config(parser.parse_args().config)
    seed_everything(config["seed"])
    organize_dataset(
        config["source_dir"],
        config["train_dir"],
        config["max_training_images_per_class"],
        seed=config["seed"],
    )
