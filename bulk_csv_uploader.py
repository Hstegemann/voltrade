# FIXED BULK CSV UPLOADER
# Compatible with all Python versions and environments
# =====================================================

import os
import pandas as pd
import json
import shutil
import hashlib
from datetime import datetime

def setup_csv_folder():
    """
    Create the CSV folder and basic setup
    """
    print("Setting up CSV upload folder...")
    
    csv_folder = "your_csv_files"
    
    # Create folder if it doesn't exist
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        print("Created folder: your_csv_files/")
    else:
        print("Folder already exists: your_csv_files/")
    
    # Check if any files are already there
    existing_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    
    if existing_files:
        print(f"Found {len(existing_files)} CSV files:")
        for f in existing_files:
            print(f"  - {f}")
        return True
    else:
        print("""
NO CSV FILES FOUND!

Please add your CSV files to the 'your_csv_files/' folder:

1. Copy your CSV files to: your_csv_files/
   Examples:
   - merged_volatility_data.csv
   - cleaned_features.csv
   - model_results.csv

2. Then run this script again

""")
        return False

def check_csv_files():
    """
    Check and validate CSV files
    """
    csv_folder = "your_csv_files"
    
    if not os.path.exists(csv_folder):
        print("Error: your_csv_files/ folder not found!")
        return []
    
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in your_csv_files/")
        return []
    
    print(f"Checking {len(csv_files)} CSV files...")
    print("-" * 50)
    
    valid_files = []
    
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder, csv_file)
        
        try:
            # Test read the file
            df = pd.read_csv(file_path, nrows=3)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            print(f"✅ {csv_file}")
            print(f"   Size: {file_size:.1f} MB")
            print(f"   Rows: {len(pd.read_csv(file_path)):,} (estimated)")
            print(f"   Columns: {len(df.columns)} - {list(df.columns)[:3]}...")
            print()
            
            valid_files.append({
                'filename': csv_file,
                'path': file_path,
                'size_mb': file_size,
                'columns': list(df.columns)
            })
            
        except Exception as e:
            print(f"❌ {csv_file} - Error: {str(e)}")
            print()
    
    print(f"Summary: {len(valid_files)} valid files found")
    return valid_files

def create_data_structure():
    """
    Create the data directory structure for Streamlit app
    """
    print("Creating data directory structure...")
    
    # Create directories
    directories = [
        "data",
        "data/shared", 
        "data/users"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}/")
        else:
            print(f"Exists: {directory}/")
    
    # Create or load metadata
    metadata_file = "data/metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print("Loaded existing metadata")
    else:
        metadata = {
            "shared_datasets": {},
            "user_datasets": {}
        }
        print("Created new metadata structure")
    
    return metadata

def upload_csv_files(valid_files, metadata):
    """
    Upload CSV files to shared datasets
    """
    if not valid_files:
        print("No valid files to upload!")
        return
    
    print(f"Uploading {len(valid_files)} files...")
    print("=" * 50)
    
    uploaded_count = 0
    
    for file_info in valid_files:
        filename = file_info['filename']
        file_path = file_info['path']
        
        print(f"Processing: {filename}")
        
        try:
            # Read the full dataset
            df = pd.read_csv(file_path)
            
            # Generate unique filename for storage
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            storage_filename = f"{filename.replace('.csv', '')}_{file_hash}.csv"
            
            # Copy to shared directory
            dest_path = os.path.join("data", "shared", storage_filename)
            shutil.copy2(file_path, dest_path)
            
            # Create dataset name
            dataset_name = filename.replace('.csv', '').replace('_', ' ').title()
            
            # Create description
            description = f"""Dataset: {dataset_name}
Original file: {filename}
Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}
Uploaded: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Auto-uploaded via bulk upload script"""
            
            # Add to metadata
            metadata["shared_datasets"][storage_filename] = {
                "name": dataset_name,
                "description": description,
                "uploader": "admin",
                "upload_date": datetime.now().isoformat(),
                "shape": [df.shape[0], df.shape[1]],
                "columns": list(df.columns),
                "file_size_mb": os.path.getsize(dest_path) / (1024 * 1024)
            }
            
            print(f"  ✅ Uploaded as: {dataset_name}")
            print(f"  📁 Saved to: {dest_path}")
            uploaded_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
        
        print()
    
    # Save metadata
    metadata_file = "data/metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 50)
    print(f"✅ Upload complete!")
    print(f"   Uploaded: {uploaded_count} files")
    print(f"   Metadata saved to: {metadata_file}")
    
    return metadata

def verify_upload(metadata):
    """
    Verify the upload worked
    """
    print("Verifying upload...")
    print("-" * 30)
    
    shared_datasets = metadata.get("shared_datasets", {})
    
    if not shared_datasets:
        print("❌ No shared datasets found!")
        return
    
    print(f"✅ Found {len(shared_datasets)} shared datasets:")
    print()
    
    for filename, info in shared_datasets.items():
        print(f"📊 {info['name']}")
        print(f"   File: {filename}")
        print(f"   Size: {info['file_size_mb']:.1f} MB")
        print(f"   Shape: {info['shape'][0]:,} × {info['shape'][1]}")
        print(f"   Date: {info['upload_date'][:10]}")
        print()
    
    print("🎉 All datasets are ready for use in the Streamlit app!")

def main():
    """
    Main function - runs the complete upload process
    """
    print("=" * 60)
    print("CSV BULK UPLOADER FOR STREAMLIT APP")
    print("=" * 60)
    print()
    
    # Step 1: Setup folder
    has_files = setup_csv_folder()
    if not has_files:
        return
    
    print()
    
    # Step 2: Check files
    valid_files = check_csv_files()
    if not valid_files:
        return
    
    print()
    
    # Step 3: Create data structure
    metadata = create_data_structure()
    
    print()
    
    # Step 4: Upload files
    metadata = upload_csv_files(valid_files, metadata)
    
    print()
    
    # Step 5: Verify
    verify_upload(metadata)
    
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("1. Start your Streamlit app: streamlit run streamlit_data_manager.py")
    print("2. Login with any demo account")
    print("3. Go to 'Browse Datasets' to see your uploaded data")
    print("4. Users can now access and analyze your datasets!")
    print("=" * 60)

def quick_add_sample_data():
    """
    Quick function to add sample data for testing
    """
    print("Creating sample data for testing...")
    
    # Setup folder
    csv_folder = "your_csv_files"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    
    # Create sample volatility data
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 09:30:00', periods=5000, freq='1min')
    
    # Sample dataset 1: Basic volatility data
    sample_data1 = pd.DataFrame({
        'DateTime': dates,
        'VIX_Close': 20 + np.cumsum(np.random.randn(5000) * 0.02),
        'SPX_Close': 4500 + np.cumsum(np.random.randn(5000) * 0.1),
        'VIXY_Close': 15 + np.cumsum(np.random.randn(5000) * 0.015),
        'Volume': np.random.randint(100000, 1000000, 5000)
    })
    
    # Make VIX negatively correlated with SPX
    sample_data1['VIX_Close'] = 25 - (sample_data1['SPX_Close'] - 4500) * 0.003 + np.random.randn(5000) * 1
    sample_data1['VIXY_Close'] = sample_data1['VIX_Close'] * 0.7 + np.random.randn(5000) * 0.5
    
    sample_data1.to_csv(os.path.join(csv_folder, 'sample_volatility_data.csv'), index=False)
    
    # Sample dataset 2: Features dataset
    sample_features = pd.DataFrame({
        'DateTime': dates[:1000],
        'VIX_return_5min': np.random.randn(1000) * 0.02,
        'VIX_return_15min': np.random.randn(1000) * 0.04,
        'VIX_RSI': np.random.uniform(20, 80, 1000),
        'VIX_MACD': np.random.randn(1000) * 0.5,
        'Volume_spike': np.random.randn(1000),
        'target_15min': np.random.choice([0, 1], 1000, p=[0.45, 0.55])
    })
    
    sample_features.to_csv(os.path.join(csv_folder, 'sample_features.csv'), index=False)
    
    print("✅ Created sample files:")
    print("  - sample_volatility_data.csv")
    print("  - sample_features.csv")
    print()
    print("Now run main() to upload them!")

# =====================================================
# RUN THE SCRIPT
# =====================================================

if __name__ == "__main__":
    # Check if user has CSV files
    if os.path.exists("your_csv_files") and os.listdir("your_csv_files"):
        # Run main upload process
        main()
    else:
        print("No CSV files found. Creating sample data...")
        quick_add_sample_data()
        print()
        print("Sample data created. Running upload process...")
        main()