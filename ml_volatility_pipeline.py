import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

# =====================================================================
# MULTI-ASSET MERGER MODULE
# =====================================================================

class MultiAssetMerger:
    """
    Merge multiple asset datasets for ML model training
    Handles alignment, missing data, and feature naming
    """
    
    def __init__(self):
        self.assets = {}
        self.merged_df = None
        self.merge_report = []
        
    def load_asset(self, file_path, asset_name, asset_type='spot'):
        """
        Load a single asset dataset
        
        Parameters:
        - file_path: path to CSV file
        - asset_name: identifier for the asset (e.g., 'SPX', 'HYG', 'VIX')
        - asset_type: 'spot' or 'futures'
        """
        print(f"\nLoading {asset_name} ({asset_type})...")
        
        try:
            # Load CSV with thousands separator handling
            df = pd.read_csv(file_path, thousands=',')
            
            # Identify datetime column (usually first column)
            date_col = df.columns[0]
            
            # Convert to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Clean up any remaining string columns that should be numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        # Remove commas and convert
                        df[col] = df[col].astype(str).str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass  # Keep as is if conversion fails
            
            # Rename columns to include asset prefix
            if asset_type == 'futures':
                # For futures, keep the detailed column names
                df.columns = [f"{col}" if asset_name in col else f"{asset_name}_{col}" 
                             for col in df.columns]
            else:
                # For spot data, add asset prefix
                df.columns = [f"{asset_name}_{col}" for col in df.columns]
            
            # Store in dictionary
            self.assets[asset_name] = {
                'data': df,
                'type': asset_type,
                'rows': len(df),
                'date_range': (df.index.min(), df.index.max()),
                'columns': list(df.columns)
            }
            
            print(f"✓ Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}")
            print(f"✓ Columns: {len(df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading {asset_name}: {e}")
            return False
    
    def analyze_time_alignment(self):
        """Analyze time alignment across all assets"""
        print("\n" + "="*60)
        print("TIME ALIGNMENT ANALYSIS")
        print("="*60)
        
        if not self.assets:
            print("No assets loaded yet!")
            return
        
        # Find overall date range
        all_starts = [info['date_range'][0] for info in self.assets.values()]
        all_ends = [info['date_range'][1] for info in self.assets.values()]
        
        overall_start = max(all_starts)
        overall_end = min(all_ends)
        
        print(f"\nOverall overlapping period:")
        print(f"  Start: {overall_start}")
        print(f"  End: {overall_end}")
        print(f"  Duration: {overall_end - overall_start}")
        
        # Show each asset's coverage
        print("\nAsset Coverage:")
        for asset_name, info in self.assets.items():
            start, end = info['date_range']
            print(f"\n{asset_name}:")
            print(f"  Start: {start}")
            print(f"  End: {end}")
            print(f"  Rows: {info['rows']:,}")
            
            # Check if asset covers the full overlap period
            if start <= overall_start and end >= overall_end:
                print(f"  ✓ Fully covers overlap period")
            else:
                if start > overall_start:
                    print(f"  ⚠️ Missing {(overall_start - start).days} days at start")
                if end < overall_end:
                    print(f"  ⚠️ Missing {(overall_end - end).days} days at end")
        
        return overall_start, overall_end
    
    def merge_all_assets(self, method='outer', freq='1min'):
        """
        Merge all loaded assets into a single DataFrame
        
        Parameters:
        - method: 'outer' (keep all timestamps) or 'inner' (only matching timestamps)
        - freq: expected frequency for resampling if needed
        """
        print("\n" + "="*60)
        print("MERGING ALL ASSETS")
        print("="*60)
        
        if not self.assets:
            print("No assets to merge!")
            return None
        
        # Show initial row counts
        print("\nInitial row counts:")
        total_unique_timestamps = set()
        for asset_name, info in self.assets.items():
            print(f"  {asset_name}: {info['rows']:,} rows")
            total_unique_timestamps.update(info['data'].index)
        
        print(f"\nTotal unique timestamps across all assets: {len(total_unique_timestamps):,}")
        
        # Start with first asset
        first_asset = list(self.assets.keys())[0]
        merged_df = self.assets[first_asset]['data'].copy()
        
        print(f"\nStarting with {first_asset}: {len(merged_df):,} rows")
        
        # Track how many timestamps we have after each merge
        merge_history = [(first_asset, len(merged_df))]
        
        # Merge each additional asset
        for asset_name in list(self.assets.keys())[1:]:
            asset_df = self.assets[asset_name]['data']
            
            print(f"\nMerging {asset_name}...")
            rows_before = len(merged_df)
            
            # Show overlap statistics before merge
            common_timestamps = merged_df.index.intersection(asset_df.index)
            print(f"  Common timestamps: {len(common_timestamps):,}")
            print(f"  Unique to current merged: {len(merged_df.index.difference(asset_df.index)):,}")
            print(f"  Unique to {asset_name}: {len(asset_df.index.difference(merged_df.index)):,}")
            
            # Merge
            merged_df = pd.merge(
                merged_df,
                asset_df,
                left_index=True,
                right_index=True,
                how=method,
                suffixes=('', f'_{asset_name}_dup')
            )
            
            rows_after = len(merged_df)
            merge_history.append((asset_name, rows_after))
            
            print(f"  Rows: {rows_before:,} → {rows_after:,} ({rows_after - rows_before:+,})")
            
            if method == 'inner' and rows_after < rows_before * 0.5:
                print(f"  ⚠️ WARNING: Lost more than 50% of rows with inner join!")
        
        # Show merge summary
        print("\n" + "-"*50)
        print("MERGE SUMMARY:")
        print("-"*50)
        for i, (asset, row_count) in enumerate(merge_history):
            if i == 0:
                print(f"Started with {asset}: {row_count:,} rows")
            else:
                prev_count = merge_history[i-1][1]
                change = row_count - prev_count
                pct_change = (change / prev_count * 100) if prev_count > 0 else 0
                print(f"After adding {asset}: {row_count:,} rows ({change:+,}, {pct_change:+.1f}%)")
        
        # Sort by datetime
        merged_df.sort_index(inplace=True)
        
        self.merged_df = merged_df
        
        print(f"\n✓ Final merged dataset: {len(merged_df):,} rows × {len(merged_df.columns)} columns")
        
        return merged_df
    
    def handle_missing_data(self, method='forward_fill', limit=5):
        """Handle missing data in merged dataset"""
        print("\n" + "="*60)
        print("HANDLING MISSING DATA")
        print("="*60)
        
        if self.merged_df is None:
            print("No merged data to process!")
            return
        
        # Analyze missing data
        missing_before = self.merged_df.isnull().sum().sum()
        total_cells = self.merged_df.shape[0] * self.merged_df.shape[1]
        
        print(f"Missing data before: {missing_before:,} ({missing_before/total_cells*100:.2f}%)")
        
        # Apply filling strategy
        if method == 'forward_fill':
            self.merged_df = self.merged_df.fillna(method='ffill', limit=limit)
        elif method == 'interpolate':
            numeric_cols = self.merged_df.select_dtypes(include=[np.number]).columns
            self.merged_df[numeric_cols] = self.merged_df[numeric_cols].interpolate(
                method='time', limit=limit
            )
        elif method == 'combined':
            # First forward fill, then interpolate
            self.merged_df = self.merged_df.fillna(method='ffill', limit=2)
            numeric_cols = self.merged_df.select_dtypes(include=[np.number]).columns
            self.merged_df[numeric_cols] = self.merged_df[numeric_cols].interpolate(
                method='time', limit=3
            )
        
        missing_after = self.merged_df.isnull().sum().sum()
        print(f"Missing data after: {missing_after:,} ({missing_after/total_cells*100:.2f}%)")
        print(f"Reduction: {(1 - missing_after/missing_before)*100:.1f}%")
        
        # Show columns with remaining missing data
        cols_with_missing = self.merged_df.isnull().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0]
        
        if len(cols_with_missing) > 0:
            print("\nColumns still containing missing data:")
            for col, count in cols_with_missing.items():
                pct = count / len(self.merged_df) * 100
                print(f"  {col}: {count:,} ({pct:.1f}%)")
    
    def create_analysis_report(self):
        """Create detailed analysis report of merged data"""
        print("\n" + "="*60)
        print("MERGED DATASET ANALYSIS REPORT")
        print("="*60)
        
        if self.merged_df is None:
            print("No merged data to analyze!")
            return
        
        df = self.merged_df
        
        # Basic info
        print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
        
        # Asset summary
        print("\nAssets Included:")
        asset_cols = {}
        for col in df.columns:
            asset_prefix = col.split('_')[0]
            if asset_prefix not in asset_cols:
                asset_cols[asset_prefix] = []
            asset_cols[asset_prefix].append(col)
        
        for asset, cols in asset_cols.items():
            print(f"  {asset}: {len(cols)} columns")
        
        # Time frequency analysis
        print("\nTime Frequency Analysis:")
        time_diffs = df.index.to_series().diff().dropna()
        freq_counts = time_diffs.value_counts().head(5)
        
        print("  Most common time intervals:")
        for interval, count in freq_counts.items():
            pct = count / len(time_diffs) * 100
            print(f"    {interval}: {count:,} ({pct:.1f}%)")
        
        # Missing data summary
        print("\nMissing Data Summary:")
        missing_by_asset = {}
        for col in df.columns:
            asset_prefix = col.split('_')[0]
            if asset_prefix not in missing_by_asset:
                missing_by_asset[asset_prefix] = 0
            missing_by_asset[asset_prefix] += df[col].isnull().sum()
        
        for asset, missing_count in missing_by_asset.items():
            if missing_count > 0:
                print(f"  {asset}: {missing_count:,} missing values")
        
        # Correlation preview
        print("\nKey Correlations (Close prices):")
        close_cols = [col for col in df.columns if 'Close' in col and 'Front' not in col and 'Next' not in col]
        if len(close_cols) > 1:
            try:
                # Only use columns with sufficient numeric data
                valid_close_cols = []
                for col in close_cols:
                    if df[col].notna().sum() > 100:  # At least 100 valid values
                        valid_close_cols.append(col)
                
                if len(valid_close_cols) > 1:
                    corr_matrix = df[valid_close_cols].corr()
                    
                    # Show interesting correlations
                    for i in range(len(valid_close_cols)):
                        for j in range(i+1, len(valid_close_cols)):
                            corr = corr_matrix.iloc[i, j]
                            if abs(corr) > 0.3 and not pd.isna(corr):  # Only show meaningful correlations
                                asset1 = valid_close_cols[i].replace('_Close', '').replace('-', '_')
                                asset2 = valid_close_cols[j].replace('_Close', '').replace('-', '_')
                                print(f"  {asset1} vs {asset2}: {corr:.3f}")
            except Exception as e:
                print(f"  Could not calculate correlations: {e}")
    
    def save_merged_dataset(self, output_path=None):
        """Save the merged dataset"""
        if self.merged_df is None:
            print("No merged data to save!")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"merged_multi_asset_dataset_{timestamp}.csv"
        
        self.merged_df.to_csv(output_path)
        print(f"\n✓ Saved merged dataset to: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
        
        return output_path

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def quick_merge_files(file_list, asset_names=None):
    """
    Quick function to merge multiple CSV files
    
    Parameters:
    - file_list: list of file paths
    - asset_names: list of asset names (if None, extracted from filenames)
    """
    merger = MultiAssetMerger()
    
    # Load each file
    for i, file_path in enumerate(file_list):
        if asset_names and i < len(asset_names):
            asset_name = asset_names[i]
        else:
            # Extract from filename
            asset_name = os.path.basename(file_path).split('_')[0]
        
        # Determine if futures or spot
        asset_type = 'futures' if 'Futures' in file_path else 'spot'
        
        merger.load_asset(file_path, asset_name, asset_type)
    
    # Analyze alignment
    merger.analyze_time_alignment()
    
    # Merge all
    merged_df = merger.merge_all_assets(method='outer')
    
    # Handle missing data
    merger.handle_missing_data(method='combined')
    
    # Create report
    merger.create_analysis_report()
    
    # Save
    output_path = merger.save_merged_dataset()
    
    return merger.merged_df, output_path

def prepare_for_ml_training(merged_df):
    """Prepare merged dataset for ML training"""
    print("\n" + "="*60)
    print("PREPARING FOR ML TRAINING")
    print("="*60)
    
    # Remove any remaining rows with too many NaN values
    threshold = 0.3  # Remove rows where >30% of values are NaN
    nan_counts = merged_df.isnull().sum(axis=1)
    max_nans = len(merged_df.columns) * threshold
    
    clean_df = merged_df[nan_counts <= max_nans].copy()
    
    print(f"Removed {len(merged_df) - len(clean_df):,} rows with excessive NaN values")
    print(f"Final dataset: {len(clean_df):,} rows × {len(clean_df.columns)} columns")
    
    # Add time-based features
    print("\nAdding time-based features...")
    clean_df['hour'] = clean_df.index.hour
    clean_df['minute'] = clean_df.index.minute
    clean_df['day_of_week'] = clean_df.index.dayofweek
    clean_df['is_regular_hours'] = ((clean_df.index.hour >= 9) & 
                                     (clean_df.index.hour < 16)).astype(int)
    
    return clean_df

# =====================================================================
# USAGE EXAMPLES
# =====================================================================

def example_usage():
    """Example of how to use the merger"""
    
    # Example 1: Manual loading
    print("EXAMPLE 1: Manual Asset Loading")
    print("-" * 60)
    
    merger = MultiAssetMerger()
    
    # Load your processed VIX futures
    merger.load_asset(
        "VIX_VIX_Futures_STRICT_ET_CONTINUOUS_30D_ENHANCED_Processed_20250701_155005.csv",
        "VIX",
        "futures"
    )
    
    # Load spot data (example paths - replace with your actual files)
    merger.load_asset("SPX_spot_data.csv", "SPX", "spot")
    merger.load_asset("HYG_spot_data.csv", "HYG", "spot")
    merger.load_asset("QQQ_spot_data.csv", "QQQ", "spot")
    
    # Load other futures
    merger.load_asset("ES_futures_processed.csv", "ES", "futures")
    
    # Analyze and merge
    merger.analyze_time_alignment()
    merged_df = merger.merge_all_assets()
    merger.handle_missing_data()
    merger.create_analysis_report()
    merger.save_merged_dataset()
    
    # Example 2: Quick merge
    print("\n\nEXAMPLE 2: Quick Merge")
    print("-" * 60)
    
    file_list = [
        "VIX_VIX_Futures_STRICT_ET_CONTINUOUS_30D_ENHANCED_Processed_20250701_155005.csv",
        "SPX_spot_data.csv",
        "HYG_spot_data.csv",
        "QQQ_spot_data.csv"
    ]
    
    merged_df, output_path = quick_merge_files(file_list)
    
    # Prepare for ML
    ml_ready_df = prepare_for_ml_training(merged_df)
    
    return ml_ready_df

# =====================================================================
# STEP-BY-STEP GUIDE
# =====================================================================

def step_by_step_merge_guide():
    """
    Interactive guide for merging datasets
    """
    print("\n" + "="*60)
    print("STEP-BY-STEP DATASET MERGING GUIDE")
    print("="*60)
    
    print("\nThis guide will help you merge your multiple asset datasets.")
    print("Make sure all your CSV files are in the current directory or provide full paths.")
    
    merger = MultiAssetMerger()
    
    # Step 1: List available files
    print("\nSTEP 1: Identifying your files")
    print("-" * 40)
    
    # Look for CSV files in current directory
    csv_files = glob.glob("*.csv")
    excel_files = glob.glob("*.xlsx")
    all_files = csv_files + excel_files
    
    print(f"Found {len(all_files)} data files in current directory:")
    for i, file in enumerate(all_files, 1):
        size_mb = os.path.getsize(file) / 1024 / 1024
        print(f"  {i}. {file} ({size_mb:.1f} MB)")
    
    # Helper function to get file by number
    def get_file_by_input(prompt, asset_name):
        while True:
            user_input = input(prompt).strip()
            
            if user_input.lower() == 'skip':
                return None
            
            # Check if it's a number
            try:
                file_num = int(user_input)
                if 1 <= file_num <= len(all_files):
                    return all_files[file_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(all_files)}")
            except ValueError:
                # Not a number, treat as file path
                if os.path.exists(user_input):
                    return user_input
                else:
                    print(f"File not found: {user_input}")
                    print("Please enter a valid file number or path, or 'skip'")
    
    # Step 2: Load files
    print("\nSTEP 2: Loading your files")
    print("-" * 40)
    print("For each asset, enter the file NUMBER from the list above (or 'skip' to skip)")
    
    files_loaded = []
    
    # Your VIX futures file
    print("\nFirst, let's identify your VIX futures file.")
    vix_file = get_file_by_input("VIX futures file number (or Enter to auto-detect): ", "VIX")
    
    if not vix_file:
        # Try to auto-detect VIX file
        vix_candidates = [f for f in all_files if 'VIX' in f.upper() and 'FUTURES' in f.upper()]
        if vix_candidates:
            vix_file = vix_candidates[0]
            print(f"Auto-detected VIX file: {vix_file}")
    
    if vix_file and os.path.exists(vix_file):
        merger.load_asset(vix_file, "VIX", "futures")
        files_loaded.append(("VIX", vix_file))
    
    # Other assets
    print("\nNow let's load your other asset files.")
    
    assets_to_load = [
        ("SPX", "spot"),
        ("HYG", "spot"),
        ("QQQ", "spot"),
        ("TLT", "spot"),
        ("ES", "futures"),
        ("DXY", "spot")
    ]
    
    for asset_name, asset_type in assets_to_load:
        file_path = get_file_by_input(f"\n{asset_name} ({asset_type}) file number: ", asset_name)
        
        if file_path and os.path.exists(file_path):
            merger.load_asset(file_path, asset_name, asset_type)
            files_loaded.append((asset_name, file_path))
        else:
            print(f"  Skipping {asset_name}")
    
    # Step 3: Analyze and merge
    print("\nSTEP 3: Analyzing and merging")
    print("-" * 40)
    
    if len(merger.assets) < 2:
        print("Need at least 2 assets to merge!")
        return None
    
    # Ask about column selection
    print("\nColumn Selection:")
    print("1. All columns (OHLCV + any extras)")
    print("2. Close prices only")
    print("3. Close prices + Volume")
    
    column_choice = input("\nSelect columns to keep (1-3, default=1): ").strip()
    
    # Filter columns based on choice
    if column_choice == '2':
        # Keep only Close columns
        print("\nFiltering to Close prices only...")
        for asset_name, asset_info in merger.assets.items():
            df = asset_info['data']
            # More precise filtering - must contain 'Close' but not be 'Unnamed'
            close_cols = [col for col in df.columns if 
                         ('Close' in col or 'close' in col) and 
                         'Unnamed' not in col and
                         not any(x in col for x in ['Open', 'High', 'Low', 'Volume'])]
            if close_cols:
                merger.assets[asset_name]['data'] = df[close_cols]
                merger.assets[asset_name]['columns'] = close_cols
                print(f"  {asset_name}: keeping {len(close_cols)} Close column(s)")
            else:
                print(f"  {asset_name}: WARNING - no Close columns found!")
    
    elif column_choice == '3':
        # Keep Close and Volume columns
        print("\nFiltering to Close + Volume columns...")
        for asset_name, asset_info in merger.assets.items():
            df = asset_info['data']
            # Keep only Close and Volume, exclude Unnamed and other OHLC
            keep_cols = [col for col in df.columns if 
                        (any(term in col for term in ['Close', 'close', 'Volume', 'volume']) and
                         'Unnamed' not in col and
                         not any(x in col for x in ['Open', 'High', 'Low']))]
            if keep_cols:
                merger.assets[asset_name]['data'] = df[keep_cols]
                merger.assets[asset_name]['columns'] = keep_cols
                print(f"  {asset_name}: keeping {len(keep_cols)} columns (Close + Volume)")
            else:
                print(f"  {asset_name}: WARNING - no Close/Volume columns found!")
    
    else:
        # Keep all columns except Unnamed
        print("\nKeeping all columns (OHLCV) - excluding Unnamed columns...")
        for asset_name, asset_info in merger.assets.items():
            df = asset_info['data']
            # Remove Unnamed columns
            valid_cols = [col for col in df.columns if 'Unnamed' not in col]
            if len(valid_cols) < len(df.columns):
                merger.assets[asset_name]['data'] = df[valid_cols]
                merger.assets[asset_name]['columns'] = valid_cols
                removed = len(df.columns) - len(valid_cols)
                print(f"  {asset_name}: removed {removed} Unnamed columns, keeping {len(valid_cols)} columns")
    
    # Analyze time alignment
    merger.analyze_time_alignment()
    
    # Ask about merge method
    print("\nMerge method:")
    print("1. Outer join (keep all timestamps from all files)")
    print("2. Inner join (only keep timestamps present in all files)")
    
    merge_choice = input("\nSelect method (1 or 2, default=1): ").strip()
    merge_method = 'inner' if merge_choice == '2' else 'outer'
    
    # Perform merge
    merged_df = merger.merge_all_assets(method=merge_method)
    
    # Step 4: Handle missing data
    print("\nSTEP 4: Handling missing data")
    print("-" * 40)
    
    print("\nMissing data handling methods:")
    print("1. Forward fill")
    print("2. Interpolation")
    print("3. Combined (forward fill then interpolate)")
    print("4. Drop rows with ANY missing data (keep only complete rows)")
    print("5. Drop rows with EXCESSIVE missing data (keep if <30% missing)")
    
    missing_choice = input("\nSelect method (1-5, default=3): ").strip()
    
    if missing_choice == '1':
        method = 'forward_fill'
        merger.handle_missing_data(method=method)
    elif missing_choice == '2':
        method = 'interpolate'
        merger.handle_missing_data(method=method)
    elif missing_choice == '4':
        # Drop any rows with missing data
        print("\nDropping rows with ANY missing data...")
        rows_before = len(merger.merged_df)
        merger.merged_df = merger.merged_df.dropna()
        rows_after = len(merger.merged_df)
        print(f"Rows: {rows_before:,} → {rows_after:,} (removed {rows_before - rows_after:,} rows)")
    elif missing_choice == '5':
        # Drop rows with excessive missing data
        print("\nDropping rows with >30% missing data...")
        rows_before = len(merger.merged_df)
        threshold = len(merger.merged_df.columns) * 0.7  # Keep if at least 70% complete
        merger.merged_df = merger.merged_df.dropna(thresh=threshold)
        rows_after = len(merger.merged_df)
        print(f"Rows: {rows_before:,} → {rows_after:,} (removed {rows_before - rows_after:,} rows)")
    else:
        method = 'combined'
        merger.handle_missing_data(method=method)
    
    # Step 5: Save results
    print("\nSTEP 5: Saving results")
    print("-" * 40)
    
    merger.create_analysis_report()
    
    output_name = input("\nOutput filename (press Enter for auto-generated): ").strip()
    output_path = merger.save_merged_dataset(output_name if output_name else None)
    
    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"\nYour merged dataset is ready: {output_path}")
    print("\nNext steps:")
    print("1. Run feature engineering on the merged dataset")
    print("2. Train ML models using the complete multi-asset data")
    
    return merger.merged_df

# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Run the interactive guide
    merged_df = step_by_step_merge_guide()
    
    if merged_df is not None:
        # Prepare for ML training
        ml_ready_df = prepare_for_ml_training(merged_df)
        
        # Save ML-ready version
        ml_ready_df.to_csv("merged_dataset_ML_ready.csv")
        print(f"\n✓ ML-ready dataset saved: merged_dataset_ML_ready.csv")