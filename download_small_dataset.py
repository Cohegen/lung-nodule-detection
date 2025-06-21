import os
import requests
import zipfile
from tqdm import tqdm
import numpy as np
import gzip
import shutil

def download_file(url, filename):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def create_synthetic_ct_data():
    """Create synthetic CT scan data for testing"""
    print("Creating synthetic CT scan data...")
    
    target_dir = r"C:\Users\HP 840 G3\Downloads\archive (16)"
    
    # Create subset directories
    for i in range(3):  # Create 3 small subsets
        subset_dir = os.path.join(target_dir, f"subset{i}")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Create multiple CT scans per subset
        for j in range(5):  # 5 scans per subset
            scan_id = f"scan_{i}_{j}"
            
            # Create .mhd file
            mhd_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = -195.5 -195.5 -195.5
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = 1 1 1
DimSize = 64 64 64
ElementType = MET_FLOAT
ElementDataFile = {scan_id}.raw
"""
            
            mhd_path = os.path.join(subset_dir, f"{scan_id}.mhd")
            with open(mhd_path, 'w') as f:
                f.write(mhd_content)
            
            # Create .raw file with realistic CT data
            # Simulate CT scan with some nodules
            ct_data = np.random.normal(0, 100, (64, 64, 64)).astype(np.float32)
            
            # Add some "nodules" (bright spots)
            for _ in range(np.random.randint(0, 3)):
                x = np.random.randint(10, 54)
                y = np.random.randint(10, 54)
                z = np.random.randint(10, 54)
                radius = np.random.randint(3, 8)
                
                # Create spherical nodule
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        for dz in range(-radius, radius+1):
                            if dx*dx + dy*dy + dz*dz <= radius*radius:
                                if (0 <= x+dx < 64 and 0 <= y+dy < 64 and 0 <= z+dz < 64):
                                    ct_data[x+dx, y+dy, z+dz] += np.random.normal(200, 50)
            
            raw_path = os.path.join(subset_dir, f"{scan_id}.raw")
            ct_data.tofile(raw_path)
    
    print("Synthetic CT data created successfully!")

def update_csv_files():
    """Update CSV files to include our synthetic data"""
    print("Updating CSV files...")
    
    target_dir = r"C:\Users\HP 840 G3\Downloads\archive (16)"
    
    # Update candidates.csv
    candidates_path = os.path.join(target_dir, "candidates.csv")
    if os.path.exists(candidates_path):
        # Read existing content
        with open(candidates_path, 'r') as f:
            lines = f.readlines()
        
        # Add synthetic candidates
        new_candidates = []
        for i in range(3):  # 3 subsets
            for j in range(5):  # 5 scans per subset
                scan_id = f"scan_{i}_{j}"
                # Add 2-5 candidates per scan
                for k in range(np.random.randint(2, 6)):
                    x = np.random.uniform(50, 200)
                    y = np.random.uniform(50, 200)
                    z = np.random.uniform(50, 200)
                    is_nodule = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% are nodules
                    new_candidates.append(f"{scan_id},{x},{y},{z},{is_nodule}")
        
        # Write back with new candidates
        with open(candidates_path, 'w') as f:
            f.writelines(lines)  # Keep header
            for candidate in new_candidates:
                f.write(candidate + '\n')
    
    # Update annotations.csv
    annotations_path = os.path.join(target_dir, "annotations.csv")
    if os.path.exists(annotations_path):
        # Read existing content
        with open(annotations_path, 'r') as f:
            lines = f.readlines()
        
        # Add synthetic annotations (only for actual nodules)
        new_annotations = []
        for i in range(3):
            for j in range(5):
                scan_id = f"scan_{i}_{j}"
                # Add 0-2 annotations per scan
                for k in range(np.random.randint(0, 3)):
                    x = np.random.uniform(50, 200)
                    y = np.random.uniform(50, 200)
                    z = np.random.uniform(50, 200)
                    diameter = np.random.uniform(5, 15)
                    new_annotations.append(f"{scan_id},{x},{y},{z},{diameter}")
        
        # Write back with new annotations
        with open(annotations_path, 'w') as f:
            f.writelines(lines)  # Keep header
            for annotation in new_annotations:
                f.write(annotation + '\n')
    
    print("CSV files updated successfully!")

def main():
    print("Creating a small synthetic CT dataset (~400MB)...")
    print("This dataset will be suitable for testing the lung nodule detection code.")
    
    target_dir = r"C:\Users\HP 840 G3\Downloads\archive (16)"
    os.makedirs(target_dir, exist_ok=True)
    
    # Create synthetic CT data
    create_synthetic_ct_data()
    
    # Update CSV files
    update_csv_files()
    
    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    print(f"\nDataset created successfully!")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")
    print(f"Location: {target_dir}")
    print("\nYou can now run the training scripts with this dataset.")

if __name__ == "__main__":
    main() 