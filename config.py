import os

class Config:
    # Dataset paths - MODIFY THESE FOR YOUR SETUP
    LUNA_ROOT = r"C:\Users\HP 840 G3\Downloads\archive (16)"
    
    # Derived paths
    ANNOTATIONS_CSV = os.path.join(LUNA_ROOT, "annotations.csv")
    CANDIDATES_CSV = os.path.join(LUNA_ROOT, "candidates.csv")
    SUBSET_GLOB = os.path.join(LUNA_ROOT, "subset*", "*.mhd")
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Adjust based on your CPU cores
    EPOCHS = 10
    
    # Model parameters
    INPUT_SIZE = (32, 48, 48)
    
    # Cache directory
    CACHE_DIR = "./cache"
    
    @classmethod
    def validate_paths(cls):
        """Validate that all required paths exist"""
        if not os.path.exists(cls.LUNA_ROOT):
            raise FileNotFoundError(f"LUNA dataset not found at: {cls.LUNA_ROOT}")
        
        if not os.path.exists(cls.ANNOTATIONS_CSV):
            raise FileNotFoundError(f"Annotations file not found: {cls.ANNOTATIONS_CSV}")
            
        if not os.path.exists(cls.CANDIDATES_CSV):
            raise FileNotFoundError(f"Candidates file not found: {cls.CANDIDATES_CSV}")
        
        # Check if any subset directories exist
        import glob
        if not glob.glob(cls.SUBSET_GLOB):
            raise FileNotFoundError(f"No subset files found matching: {cls.SUBSET_GLOB}")
        
        print("✓ All dataset paths validated successfully")

if __name__ == "__main__":
    samples = findPositiveSamples(limit=1)
    if samples:
        showCandidate(samples[0].series_uid)