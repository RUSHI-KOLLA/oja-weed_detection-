"""
Download and prepare all 3 datasets for AgroKD-Net
Datasets: MH-Weed16, CottonWeed, RiceWeed
Compatible with Google Colab
"""
import os
import subprocess
import shutil
from pathlib import Path

def mount_drive():
    """Mount Google Drive in Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
        return True
    except ImportError:
        print("⚠️ Not running in Colab, using local storage")
        return False

def create_dataset_structure(base_path, dataset_name):
    """Create standardized folder structure"""
    for split in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            path = os.path.join(base_path, dataset_name, folder, split)
            os.makedirs(path, exist_ok=True)
    print(f"📁 Created folder structure for {dataset_name}")

def download_mhweed16(base_path):
    """Instructions for MH-Weed16 download"""
    print("\n🌿 MH-Weed16 Dataset (Maharashtra Weed Dataset — PRIMARY):")
    print("  📥 Option 1 - Kaggle (recommended):")
    print('     kaggle datasets download -s "MH-Weed16"')
    print('     # Or search "Maharashtra weed dataset" at https://www.kaggle.com/datasets')
    print("  📥 Option 2 - Direct search:")
    print('     Search "MH-Weed16 Maharashtra weed detection" for dataset repositories')
    print("  ℹ️  ~25,972 images, 16 weed species, bounding box annotations (YOLO format)")
    create_dataset_structure(base_path, "mhweed16")

def download_cottonweed(base_path):
    """Instructions for CottonWeed download"""
    print("\n🌾 CottonWeed Dataset:")
    print("  📥 Option 1 - Roboflow (recommended):")
    print('     pip install roboflow')
    print('     from roboflow import Roboflow')
    print('     rf = Roboflow(api_key="YOUR_API_KEY")')
    print('     project = rf.workspace().project("cottonweeddet12")')
    print('     dataset = project.version(1).download("yolov8")')
    print("  📥 Option 2 - Kaggle:")
    print('     kaggle datasets download -d yuzhenlu/cottonweeddet12')
    create_dataset_structure(base_path, "cottonweed")

def download_riceweed(base_path):
    """Instructions for RiceWeed download"""
    print("\n🌾 RiceWeed Dataset:")
    print("  📥 Option 1 - Roboflow:")
    print('     Search "rice weed detection" at https://universe.roboflow.com')
    print('     Export in YOLOv8 format')
    print("  📥 Option 2 - Kaggle:")
    print('     Search "paddy weed detection" at https://www.kaggle.com/datasets')
    create_dataset_structure(base_path, "riceweed")

def count_images(base_path, dataset_name):
    """Count images per split"""
    print(f"\n📊 Image count for {dataset_name}:")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_path, dataset_name, 'images', split)
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}: {count} images")
        else:
            print(f"  {split}: directory not found")

if __name__ == '__main__':
    print("🌱 AgroKD-Net Dataset Downloader")
    print("=" * 50)
    
    is_colab = mount_drive()
    base_path = "/content/drive/MyDrive/AgroKD_Datasets" if is_colab else "datasets"
    os.makedirs(base_path, exist_ok=True)
    
    download_mhweed16(base_path)
    download_cottonweed(base_path)
    download_riceweed(base_path)
    
    print("\n" + "=" * 50)
    print("✅ Dataset setup complete!")
    print(f"📁 Base path: {base_path}")
    print("\nFinal 3 datasets:")
    print("  1. MH-Weed16 🇮🇳 - Detection (16 Indian weed species, ~25,972 images) [PRIMARY]")
    print("  2. CottonWeed 🇮🇳🇺🇸 - Detection (15 weeds + cotton)")
    print("  3. RiceWeed 🇮🇳 - Detection (rice + weeds in paddy fields)")
