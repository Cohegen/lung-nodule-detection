import os
import numpy as np

def create_small_dataset():
    """Create a small synthetic CT dataset (~400MB)"""
    print("Creating small synthetic CT dataset...")
    
    target_dir = r"C:\Users\HP 840 G3\Downloads\archive (16)"
    scan_dim = 128
    
    # Create subset directories
    for i in range(2):  # 2 subsets
        subset_dir = os.path.join(target_dir, f"subset{i}")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Create CT scans per subset
        for j in range(10):  # 10 scans per subset
            scan_id = f"scan_{i}_{j}"
            
            # Create .mhd file
            mhd_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = 1 1 1
DimSize = {scan_dim} {scan_dim} {scan_dim}
ElementType = MET_FLOAT
ElementDataFile = {scan_id}.raw
"""
            
            mhd_path = os.path.join(subset_dir, f"{scan_id}.mhd")
            with open(mhd_path, 'w') as f:
                f.write(mhd_content)
            
            # Create .raw file with realistic CT data
            ct_data = np.random.normal(0, 100, (scan_dim, scan_dim, scan_dim)).astype(np.float32)
            
            # Add some "nodules" (bright spots)
            for _ in range(np.random.randint(0, 5)):
                x = np.random.randint(25, scan_dim-25)
                y = np.random.randint(25, scan_dim-25)
                z = np.random.randint(25, scan_dim-25)
                radius = np.random.randint(5, 12)
                
                # Create spherical nodule
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        for dz in range(-radius, radius+1):
                            if dx*dx + dy*dy + dz*dz <= radius*radius:
                                if (0 <= x+dx < scan_dim and 0 <= y+dy < scan_dim and 0 <= z+dz < scan_dim):
                                    ct_data[x+dx, y+dy, z+dz] += np.random.normal(300, 100)
            
            raw_path = os.path.join(subset_dir, f"{scan_id}.raw")
            ct_data.tofile(raw_path)
    
    # Overwrite candidates.csv with only synthetic, in-bounds data
    candidates_path = os.path.join(target_dir, "candidates.csv")
    new_candidates = []
    for i in range(2):
        for j in range(10):
            scan_id = f"scan_{i}_{j}"
            for k in range(np.random.randint(3, 8)):
                x = np.random.uniform(25, scan_dim-25-1e-3)
                y = np.random.uniform(25, scan_dim-25-1e-3)
                z = np.random.uniform(25, scan_dim-25-1e-3)
                is_nodule = np.random.choice([0, 1], p=[0.7, 0.3])
                new_candidates.append(f"{scan_id},{x},{y},{z},{is_nodule}")
    with open(candidates_path, 'w') as f:
        f.write("seriesuid,coordX,coordY,coordZ,class\n")
        for candidate in new_candidates:
            f.write(candidate + '\n')
    
    # Overwrite annotations.csv with only synthetic, in-bounds data
    annotations_path = os.path.join(target_dir, "annotations.csv")
    new_annotations = []
    for i in range(2):
        for j in range(10):
            scan_id = f"scan_{i}_{j}"
            for k in range(np.random.randint(0, 4)):
                x = np.random.uniform(25, scan_dim-25-1e-3)
                y = np.random.uniform(25, scan_dim-25-1e-3)
                z = np.random.uniform(25, scan_dim-25-1e-3)
                diameter = np.random.uniform(5, 20)
                new_annotations.append(f"{scan_id},{x},{y},{z},{diameter}")
    with open(annotations_path, 'w') as f:
        f.write("seriesuid,coordX,coordY,coordZ,diameter_mm\n")
        for annotation in new_annotations:
            f.write(annotation + '\n')
    
    # Calculate size
    total_size = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    print(f"Dataset created! Size: {total_size / (1024*1024):.1f} MB")
    print("Ready for training!")

if __name__ == "__main__":
    create_small_dataset() 