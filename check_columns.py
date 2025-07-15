import pandas as pd
import numpy as np

def check_columns_and_fix_target():
    """Check what columns are available and fix the target variable issue"""
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv('merged_multi_asset_dataset_20250714_112740.csv')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nAll columns ({len(df.columns)}):")
    print("="*60)
    
    # Group columns by asset
    column_groups = {}
    
    for col in sorted(df.columns):
        print(f"  {col}")
        
        # Try to identify asset name
        if '_' in col:
            parts = col.split('_')
            asset = parts[0]
            if asset not in column_groups:
                column_groups[asset] = []
            column_groups[asset].append(col)
    
    print("\n" + "="*60)
    print("COLUMNS GROUPED BY ASSET:")
    print("="*60)
    
    for asset, cols in sorted(column_groups.items()):
        print(f"\n{asset}: {len(cols)} columns")
        for col in cols[:5]:  # Show first 5
            print(f"  - {col}")
        if len(cols) > 5:
            print(f"  ... and {len(cols)-5} more")
    
    # Look for Close columns specifically
    print("\n" + "="*60)
    print("CLOSE PRICE COLUMNS:")
    print("="*60)
    
    close_columns = [col for col in df.columns if 'Close' in col or 'close' in col]
    for col in close_columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")
    
    # Check for VIXY specifically (case-insensitive)
    print("\n" + "="*60)
    print("SEARCHING FOR VIXY OR SIMILAR:")
    print("="*60)
    
    vixy_candidates = []
    for col in df.columns:
        if any(x in col.upper() for x in ['VIXY', 'VXX', 'UVXY', 'SVXY', 'SVIX']):
            vixy_candidates.append(col)
            print(f"  Found: {col}")
    
    if not vixy_candidates:
        print("  No VIXY-related columns found!")
        print("\n  Looking for other volatility ETFs...")
        
        # Look for any volatility-related columns
        vol_keywords = ['VIX', 'VOL', 'VX']
        for keyword in vol_keywords:
            matches = [col for col in df.columns if keyword in col.upper()]
            if matches:
                print(f"\n  {keyword} columns: {matches[:5]}")
    
    # Let's check what the actual data looks like
    print("\n" + "="*60)
    print("SAMPLE DATA:")
    print("="*60)
    print(df.head())
    
    return df, close_columns

# Run the check
df, close_cols = check_columns_and_fix_target()

# Now let's create a flexible target variable fixer
class FlexibleTargetFixer:
    """Works with whatever volatility-related columns are available"""
    
    def __init__(self, df):
        self.df = df
        self.identified_assets = {}
        
    def identify_tradeable_assets(self):
        """Identify what we can actually trade"""
        
        print("\n" + "="*60)
        print("IDENTIFYING TRADEABLE ASSETS")
        print("="*60)
        
        # Look for close price columns
        close_cols = [col for col in self.df.columns if 'Close' in col]
        
        # Group by asset
        for col in close_cols:
            # Extract asset name (everything before _Close)
            asset = col.replace('_Close', '')
            
            # Check if this looks like a volatility product
            if any(x in asset.upper() for x in ['VIX', 'VX', 'VOL', 'VIXY', 'SVIX', 'UVXY']):
                self.identified_assets[asset] = {
                    'close_col': col,
                    'type': 'volatility'
                }
            elif any(x in asset.upper() for x in ['SPX', 'SPY', 'ES']):
                self.identified_assets[asset] = {
                    'close_col': col,
                    'type': 'equity'
                }
            else:
                self.identified_assets[asset] = {
                    'close_col': col,
                    'type': 'other'
                }
        
        print("\nIdentified assets:")
        for asset, info in self.identified_assets.items():
            non_null = self.df[info['close_col']].notna().sum()
            print(f"  {asset} ({info['type']}): {non_null:,} price points")
        
        # Find the best volatility product to trade
        vol_assets = {k: v for k, v in self.identified_assets.items() if v['type'] == 'volatility'}
        
        if vol_assets:
            # Pick the one with most data
            best_vol = max(vol_assets.keys(), key=lambda x: self.df[vol_assets[x]['close_col']].notna().sum())
            print(f"\n✅ Selected volatility asset for trading: {best_vol}")
            return best_vol, vol_assets[best_vol]['close_col']
        else:
            print("\n❌ No volatility assets found! Will use VIX if available...")
            if 'VIX' in self.identified_assets:
                return 'VIX', self.identified_assets['VIX']['close_col']
            else:
                return None, None
    
    def create_target_variables(self, asset_name, close_col):
        """Create target variables for any asset"""
        
        if not close_col:
            print("ERROR: No tradeable asset found!")
            return None
            
        print(f"\n" + "="*60)
        print(f"CREATING TARGET VARIABLES FOR {asset_name}")
        print("="*60)
        
        prices = self.df[close_col].dropna()
        print(f"Using {len(prices):,} price points")
        
        # Create a results dataframe
        features_df = pd.DataFrame(index=self.df.index)
        
        # Create forward-looking targets
        horizons = [5, 15, 30, 60]
        
        for horizon in horizons:
            # Future price
            future_price = self.df[close_col].shift(-horizon)
            current_price = self.df[close_col]
            
            # Continuous returns
            ret_col = f'{asset_name}_return_next_{horizon}min'
            features_df[ret_col] = (future_price - current_price) / current_price
            
            # Binary direction
            dir_col = f'{asset_name}_up_next_{horizon}min'
            features_df[dir_col] = (future_price > current_price).astype(float)
            
            # Calculate statistics
            valid_mask = features_df[dir_col].notna()
            if valid_mask.sum() > 0:
                pct_up = features_df.loc[valid_mask, dir_col].mean() * 100
                print(f"\n{horizon}-min forward returns:")
                print(f"  Valid samples: {valid_mask.sum():,}")
                print(f"  Up moves: {pct_up:.1f}%")
                print(f"  Down moves: {100-pct_up:.1f}%")
        
        return features_df
    
    def create_all_features(self, target_asset, target_col):
        """Create all features based on available columns"""
        
        print(f"\n" + "="*60)
        print("CREATING PREDICTOR FEATURES")
        print("="*60)
        
        features_df = pd.DataFrame(index=self.df.index)
        
        # For each identified asset, create features
        for asset, info in self.identified_assets.items():
            close_col = info['close_col']
            prices = self.df[close_col]
            
            print(f"\nCreating features for {asset}...")
            
            # Lagged returns
            for lag in [5, 10, 15, 30, 60]:
                features_df[f'{asset}_return_lag{lag}'] = prices.pct_change(lag)
            
            # Simple technical indicators
            features_df[f'{asset}_SMA_20'] = prices.rolling(20).mean()
            features_df[f'{asset}_price_to_SMA20'] = prices / features_df[f'{asset}_SMA_20']
            features_df[f'{asset}_volatility_30'] = prices.pct_change().rolling(30).std()
        
        print(f"\nTotal features created: {len(features_df.columns)}")
        
        return features_df

# Use the flexible fixer
print("\n" + "="*60)
print("USING FLEXIBLE TARGET FIXER")
print("="*60)

fixer = FlexibleTargetFixer(df)
target_asset, target_col = fixer.identify_tradeable_assets()

if target_asset:
    # Create target variables
    target_df = fixer.create_target_variables(target_asset, target_col)
    
    # Create features
    features_df = fixer.create_all_features(target_asset, target_col)
    
    # Combine everything
    final_df = pd.concat([df, features_df, target_df], axis=1)
    
    # Save the result
    output_file = f'fixed_features_{target_asset}.csv'
    final_df.to_csv(output_file)
    print(f"\n✅ Fixed features saved to: {output_file}")
    print(f"Target asset: {target_asset}")
    print(f"Ready for Phase 4: Model Development!")
else:
    print("\n❌ Could not identify any tradeable volatility assets!")
    print("Please check your data file and column names.")