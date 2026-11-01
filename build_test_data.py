# DEPRECATED: This script has been superseded by scripts/ingest_data.py (DVC Stage 0).
# It is kept for historical reference only. Do NOT run it directly — it shares the same
# source_dir and seed as dataset_organizer.py without a cross-script exclusion list,
# risking train/test raw-source overlap.
# Use:  dvc repro ingest_data
import os
import random
import argparse
from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.reproducibility import seed_everything
from PIL import Image
import torchvision.transforms as transforms

def build_testing_dataset(config_path="configs/data.yaml"):
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    rng = random.Random(seed)
    # -----------------------------
    # 1. Configuration
    # -----------------------------
    # Change this to the folder where your downloaded ImageNet synset folders are stored
    SOURCE_DIR = config["source_dir"]
    # The new folder where your test dataset will be generated
    TEST_DIR = config["test_dir"]
    if os.path.isdir(TEST_DIR) and any(os.scandir(TEST_DIR)):
        raise FileExistsError(
            f"Refusing to overwrite or mix a fixed evaluation dataset at {TEST_DIR}; "
            "select a new output directory in the data config."
        )
    CLASS_MAPPING = {
        "Bus": ["n04487081", "n04146614", "n02924116"],
        "Truck": ["n04467665", "n03417042", "n03345487"],
        "Car": [
            "n02958343", "n03100232", "n03594945", 
            "n03769881", "n04037443", "n04285008",
        ],
        "Bike": ["n03790512", "n02835271"],
        "None": [
            "n06874185", # Traffic light
            "n04328186", # Street sign
            "n03881040", # Park bench
            "n03710193"  # Mailbox
        ]
    }
    # -----------------------------
    # 2. Augmentation Pipeline for Test Data
    # -----------------------------
    # We apply the visual corruptions and occlusions, but NOT normalization
    corruption = config["corruption"]
    image_size = tuple(config["image_size"])
    test_augmentation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=corruption["horizontal_flip_probability"]),
        transforms.RandomAffine(
            degrees=corruption["rotation_degrees"],
            translate=tuple(corruption["translation_fraction"]),
            scale=tuple(corruption["scale_range"]),
        ),
        transforms.ColorJitter(
            brightness=corruption["brightness_jitter"],
            contrast=corruption["contrast_jitter"],
        ),
        transforms.ToTensor(), # Required for RandomErasing
        transforms.RandomErasing(
            p=corruption["random_erasing_probability"],
            scale=tuple(corruption["random_erasing_scale"]),
            ratio=tuple(corruption["random_erasing_ratio"]),
            value=0,
        ),
        transforms.ToPILImage() # Convert back to save as JPEG
    ])

    # Basic resize for the "clean" half of the dataset
    clean_transform = transforms.Compose([
        transforms.Resize(image_size)
    ])

    # -----------------------------
    # 3. Dataset Generation Loop
    # -----------------------------
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    for class_name, synsets in CLASS_MAPPING.items():
        print(f"Processing Test Class: {class_name}...")
        
        class_dest_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(class_dest_dir, exist_ok=True)
        
        # Gather all valid images from the synsets belonging to this class
        all_image_paths = []
        for synset in synsets:
            synset_path = os.path.join(SOURCE_DIR, synset)
            if os.path.exists(synset_path):
                for img_name in os.listdir(synset_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(synset_path, img_name))
            else:
                print(f"  Warning: Synset folder {synset} not found in {SOURCE_DIR}")
        
        # Shuffle to ensure randomness
        rng.shuffle(all_image_paths)
        
        # Grab up to 500 images
        selected_images = all_image_paths[: config["images_per_class"]]
        
        if len(selected_images) < config["images_per_class"]:
            print(
                f"  Note: Only found {len(selected_images)} images for {class_name} "
                f"(Requested {config['images_per_class']})."
            )
            
        # Split half and half
        half_point = len(selected_images) // 2
        clean_images = selected_images[:half_point]
        aug_images = selected_images[half_point:]
        
        # 1. Process and save the CLEAN images (250 images)
        for i, img_path in enumerate(clean_images):
            try:
                img = Image.open(img_path).convert("RGB")
                img = clean_transform(img)
                save_path = os.path.join(class_dest_dir, f"clean_{i:04d}.jpg")
                img.save(save_path)
            except Exception as e:
                print(f"  Error processing clean image {img_path}: {e}")
                
        # 2. Process and save the AUGMENTED images (250 images)
        for i, img_path in enumerate(aug_images):
            try:
                img = Image.open(img_path).convert("RGB")
                img = test_augmentation(img)
                save_path = os.path.join(class_dest_dir, f"aug_{i:04d}.jpg")
                img.save(save_path)
            except Exception as e:
                print(f"  Error processing augmented image {img_path}: {e}")

    print(f"\nTesting dataset successfully generated at: {TEST_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    build_testing_dataset(parser.parse_args().config)
