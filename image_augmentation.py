import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import os
import random
import numpy as np
from pathlib import Path
import time

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "-", torch.cuda.get_device_name(0) if device=="cuda" else "")

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_augmentation_transforms():
    """Create a comprehensive set of augmentation transforms"""
    return [
        # Geometric transformations
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # Color and brightness transformations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        
        # Noise and blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.RandomInvert(p=0.1),
        
        # Perspective transformation
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        
        # Elastic transformation (simulated with random affine)
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=(-10, 10)),
    ]

def apply_augmentations(image, transforms_list, num_augmentations=5):
    """Apply multiple augmentations to a single image"""
    augmented_images = []
    
    for i in range(num_augmentations):
        # Randomly select 2-3 transforms to apply
        num_transforms = random.randint(2, 3)
        selected_transforms = random.sample(transforms_list, num_transforms)
        
        # Apply selected transforms
        augmented_img = image
        for transform in selected_transforms:
            augmented_img = transform(augmented_img)
        
        augmented_images.append(augmented_img)
    
    return augmented_images

def process_images(input_folder, output_folder, num_augmentations=5):
    """Process all images in input folder and save augmented versions"""
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(input_folder) 
                   if Path(f).suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # Get augmentation transforms
    augmentation_transforms = get_augmentation_transforms()
    
    total_images = 0
    start_time = time.time()
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        # Load image
        image_path = os.path.join(input_folder, image_file)
        try:
            original_image = Image.open(image_path).convert('RGB')
            
            # Generate augmented versions
            augmented_images = apply_augmentations(original_image, augmentation_transforms, num_augmentations)
            
            # Save original image to output folder
            original_output_path = os.path.join(output_folder, f"original_{image_file}")
            original_image.save(original_output_path)
            
            # Save augmented images
            for j, aug_img in enumerate(augmented_images):
                aug_filename = f"aug_{j+1}_{image_file}"
                aug_output_path = os.path.join(output_folder, aug_filename)
                aug_img.save(aug_output_path)
                total_images += 1
            
            total_images += 1  # Count original image
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total images generated: {total_images}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Images saved to: {output_folder}")
    
    return total_images

if __name__ == "__main__":
    # Configuration
    input_folder = "o-set-1"
    output_folder = "o-set-1-aug"
    num_augmentations_per_image = 5
    
    print("Starting image augmentation pipeline...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Augmentations per image: {num_augmentations_per_image}")
    
    # Process images
    total_generated = process_images(input_folder, output_folder, num_augmentations_per_image)
    
    print(f"\nPipeline completed successfully!")
    print(f"Generated {total_generated} total images (original + augmented)")
    print(f"Check the '{output_folder}' folder for results")
