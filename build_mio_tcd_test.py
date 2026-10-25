import os
import random
from vehicle_classifier_core.reproducibility import seed_everything
from PIL import Image
import torchvision.transforms as transforms

def build_mio_tcd_dataset(seed=1729):
    seed_everything(seed)
    rng = random.Random(seed)
    # -----------------------------
    # 1. Configuration
    # -----------------------------
    # Path where you extracted the Kaggle MIO-TCD train images
    SOURCE_DIR = "mio_tcd_raw/train1" 
    # The new mapped and balanced folder
    TEST_DIR = "./test_MIO_TCD_data"   
    if os.path.isdir(TEST_DIR) and any(os.scandir(TEST_DIR)):
        raise FileExistsError(
            f"Refusing to overwrite or mix a fixed OOD dataset at {TEST_DIR}; choose a new output path."
        )

    # Mapping the 11 Kaggle classes to your 5 assignment classes
    MIO_CLASS_MAPPING = {
        "Bus": ["bus"],
        "Truck": ["articulated_truck", "pickup_truck", "single_unit_truck"],
        "Car": ["car", "work_van"],
        "Bike": ["bicycle", "motorcycle"],
        "None": ["pedestrian", "background"]
    }

    # -----------------------------
    # 2. Augmentation Pipeline
    # -----------------------------
    # Same strict evaluation pipeline: 50% clean, 50% heavily augmented
    test_augmentation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(), 
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        transforms.ToPILImage() 
    ])

    clean_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    # -----------------------------
    # 3. Dataset Generation Loop
    # -----------------------------
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    for target_class, source_folders in MIO_CLASS_MAPPING.items():
        print(f"Mapping to Target Class: {target_class}...")
        
        class_dest_dir = os.path.join(TEST_DIR, target_class)
        os.makedirs(class_dest_dir, exist_ok=True)
        
        # Gather all valid images from the mapped MIO-TCD folders
        all_image_paths = []
        for folder in source_folders:
            folder_path = os.path.join(SOURCE_DIR, folder)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(folder_path, img_name))
            else:
                print(f"  Warning: Kaggle folder '{folder}' not found in {SOURCE_DIR}")
        
        # Shuffle to ensure we get a random mix (e.g., mixing motorcycles and bicycles evenly)
        rng.shuffle(all_image_paths)
        
        # Grab exactly 500 images per class to keep the test balanced
        selected_images = all_image_paths[:500]
        
        if len(selected_images) < 500:
            print(f"  Note: Only found {len(selected_images)} images for {target_class}.")
            
        half_point = len(selected_images) // 2
        clean_images = selected_images[:half_point]
        aug_images = selected_images[half_point:]
        
        # 1. Process CLEAN images
        for i, img_path in enumerate(clean_images):
            try:
                img = Image.open(img_path).convert("RGB")
                img = clean_transform(img)
                save_path = os.path.join(class_dest_dir, f"clean_{i:04d}.jpg")
                img.save(save_path)
            except Exception as e:
                pass # Skip corrupted Kaggle images quietly
                
        # 2. Process AUGMENTED images
        for i, img_path in enumerate(aug_images):
            try:
                img = Image.open(img_path).convert("RGB")
                img = test_augmentation(img)
                save_path = os.path.join(class_dest_dir, f"aug_{i:04d}.jpg")
                img.save(save_path)
            except Exception as e:
                pass

    print(f"\nMIO-TCD Stress-Test dataset successfully generated at: {TEST_DIR}")

if __name__ == "__main__":
    build_mio_tcd_dataset(seed=1729)
