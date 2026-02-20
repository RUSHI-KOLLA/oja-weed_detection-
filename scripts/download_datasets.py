"""
Download and prepare all 3 datasets for AgroKD-Net
Datasets: CottonWeed, DeepWeeds, RiceWeed
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

def download_deepweeds(base_path):
    """Download DeepWeeds from GitHub"""
    print("\n🌿 Downloading DeepWeeds...")
    dest = os.path.join(base_path, "deepweeds")
    if os.path.exists(os.path.join(dest, "images")):
        print("  ✅ DeepWeeds already downloaded")
        return
    try:
        subprocess.run(["git", "clone", "https://github.com/AlexOlsen/DeepWeeds.git",
                         os.path.join(base_path, "DeepWeeds_raw")], check=True)
        print("  ✅ DeepWeeds cloned successfully")
    except subprocess.CalledProcessError as exc:
        print(f"  ❌ git clone failed: {exc}")
        print("  💡 Ensure git is installed and you have internet access, then retry.")
        return
    print("  ⚠️ Note: This is a CLASSIFICATION dataset - needs adaptation for detection")

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
    
    download_deepweeds(base_path)
    download_cottonweed(base_path)
    download_riceweed(base_path)
    
    print("\n" + "=" * 50)
    print("✅ Dataset setup complete!")
    print(f"📁 Base path: {base_path}")
    print("\nFinal 3 datasets:")
    print("  1. CottonWeed 🇮🇳🇺🇸 - Detection (15 weeds + cotton)")
    print("  2. DeepWeeds 🇦🇺 - Classification (8 weeds + negative)")
    print("  3. RiceWeed 🇮🇳 - Detection (rice + weeds in paddy fields)")
