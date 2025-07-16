import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class VIXFuturesSimplifier:
    """
    Simplify VIX futures data into continuous contracts and spot equivalent
    Starting from the very beginning!
    """
    
    def __init__(self):
        self.contract_months = {
            'F': 1,  # January
            'G': 2,  # February  
            'H': 3,  # March
            'I': 4,  # April (skipped)
            'J': 4,  # April
            'K': 5,  # May
            'M': 6,  # June
            'N': 7,  # July
            'Q': 8,  # August
            'U': 9,  # September
            'V': 10, # October
            'X': 11, # November
            'Z': 12  # December
        }
        
    def analyze_current_data(self, filepath='merged_multi_asset_dataset_20250714_112740.csv'):
        """Step 1: Understand what we have"""
        
        print("="*60)
        print("STEP 1: ANALYZING YOUR CURRENT DATA")
        print("="*60)
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"\nLoaded data with shape: {df.shape}")
        
        # Parse date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Identify VIX futures columns
        vix_futures = {}
        for col in df.columns:
            if 'VIX_Close' in col and '-' in col:
                # Extract contract code (e.g., VIF25 from VIX_Close-VIF25)
                contract = col.split('-')[1]
                vix_futures[contract] = col
        
        print(f"\nFound {len(vix_futures)} VIX futures contracts:")
        for contract, col in sorted(vix_futures.items()):
            non_null = df[col].notna().sum()
            print(f"  {contract}: {non_null:,} data points")
        
        # Check other assets
        other_assets = []
        for col in df.columns:
            if 'Close' in col and 'VIX' not in col:
                other_assets.append(col)
        
        print(f"\nOther assets found:")
        for col in other_assets[:10]:
            print(f"  {col}")
        
        self.df = df
        self.vix_futures = vix_futures
        
        return df
    
    def create_continuous_contract(self):
        """Step 2: Create a continuous VIX futures contract"""
        
        print("\n" + "="*60)
        print("STEP 2: CREATING CONTINUOUS VIX CONTRACT")
        print("="*60)
        
        # We'll create a simple continuous contract by:
        # 1. Always using the front month contract
        # 2. Rolling to the next month when appropriate
        
        continuous_prices = pd.Series(index=self.df.index, dtype=float)
        
        # For each date, find the appropriate contract
        for date in self.df.index:
            current_month = date.month
            current_year = date.year
            
            # Find the front month contract (next expiring)
            best_contract = None
            best_price = None
            
            for contract, col in self.vix_futures.items():
                # Parse contract month and year
                if len(contract) >= 4:  # e.g., VIF25
                    month_code = contract[2]  # F
                    year = int('20' + contract[3:5])  # 25 -> 2025
                    
                    if month_code in self.contract_months:
                        contract_month = self.contract_months[month_code]
                        
                        # Is this contract still valid (not expired)?
                        if (year > current_year) or (year == current_year and contract_month >= current_month):
                            price = self.df.loc[date, col]
                            if pd.notna(price) and price > 0:
                                if best_contract is None or (year, contract_month) < (int('20' + best_contract[3:5]), self.contract_months[best_contract[2]]):
                                    best_contract = contract
                                    best_price = price
            
            if best_price is not None:
                continuous_prices[date] = best_price
        
        # Remove any NaN values
        continuous_prices = continuous_prices.dropna()
        
        print(f"Created continuous contract with {len(continuous_prices):,} data points")
        print(f"Price range: ${continuous_prices.min():.2f} to ${continuous_prices.max():.2f}")
        
        self.continuous_vix = continuous_prices
        
        # Visualize
        plt.figure(figsize=(12, 6))
        continuous_prices.plot()
        plt.title('VIX Continuous Contract')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return continuous_prices
    
    def create_simple_dataset(self):
        """Step 3: Create a simplified dataset with key assets"""
        
        print("\n" + "="*60)
        print("STEP 3: CREATING SIMPLIFIED DATASET")
        print("="*60)
        
        # Create new simplified dataframe
        simple_df = pd.DataFrame(index=self.continuous_vix.index)
        
        # Add continuous VIX
        simple_df['VIX_Close'] = self.continuous_vix
        
        # Add SPX if available
        spx_col = next((col for col in self.df.columns if 'SPX_Close' in col), None)
        if spx_col:
            simple_df['SPX_Close'] = self.df.loc[simple_df.index, spx_col]
            print("Added SPX data")
        
        # Add HYG if available
        hyg_col = next((col for col in self.df.columns if 'HYG_Close' in col), None)
        if hyg_col:
            simple_df['HYG_Close'] = self.df.loc[simple_df.index, hyg_col]
            print("Added HYG data")
        
        # Add other important assets
        for asset in ['TLT', 'QQQ', 'DXY']:
            asset_col = next((col for col in self.df.columns if f'{asset}_Close' in col), None)
            if asset_col:
                simple_df[f'{asset}_Close'] = self.df.loc[simple_df.index, asset_col]
                print(f"Added {asset} data")
        
        print(f"\nSimplified dataset shape: {simple_df.shape}")
        print(f"Columns: {list(simple_df.columns)}")
        
        # Check data completeness
        print("\nData completeness:")
        for col in simple_df.columns:
            completeness = simple_df[col].notna().sum() / len(simple_df) * 100
            print(f"  {col}: {completeness:.1f}%")
        
        self.simple_df = simple_df
        
        return simple_df
    
    def add_vixy_equivalent(self):
        """Step 4: Create VIXY-equivalent from VIX futures"""
        
        print("\n" + "="*60)
        print("STEP 4: CREATING VIXY EQUIVALENT")
        print("="*60)
        
        # VIXY tracks the S&P 500 VIX Short-Term Futures Index
        # This is approximately a weighted average of the first and second month VIX futures
        # For simplicity, we'll use the continuous contract with some adjustments
        
        # Calculate daily returns of VIX futures
        vix_returns = self.continuous_vix.pct_change()
        
        # VIXY has decay due to contango, approximately -5% to -10% per month
        # This is about -0.25% per day
        daily_decay = -0.0025
        
        # Create synthetic VIXY starting at a reasonable price
        vixy_prices = pd.Series(index=self.simple_df.index, dtype=float)
        vixy_prices.iloc[0] = 20.0  # Starting price
        
        # Calculate VIXY prices including decay
        for i in range(1, len(vixy_prices)):
            if pd.notna(vix_returns.iloc[i]):
                # VIXY return = VIX futures return + decay
                vixy_return = vix_returns.iloc[i] * 0.5 + daily_decay  # 0.5 factor because VIXY is less volatile
                vixy_prices.iloc[i] = vixy_prices.iloc[i-1] * (1 + vixy_return)
            else:
                vixy_prices.iloc[i] = vixy_prices.iloc[i-1]
        
        # Add to dataframe
        self.simple_df['VIXY_Close'] = vixy_prices
        
        print(f"Created synthetic VIXY with {len(vixy_prices):,} data points")
        print(f"VIXY price range: ${vixy_prices.min():.2f} to ${vixy_prices.max():.2f}")
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # VIX vs VIXY
        ax1.plot(self.simple_df.index, self.simple_df['VIX_Close'], label='VIX', alpha=0.7)
        ax1.set_ylabel('VIX Level')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.simple_df.index, self.simple_df['VIXY_Close'], label='VIXY (Synthetic)', color='green', alpha=0.7)
        ax2.set_ylabel('VIXY Price')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return self.simple_df
    
    def save_simplified_data(self):
        """Step 5: Save the simplified dataset"""
        
        print("\n" + "="*60)
        print("STEP 5: SAVING SIMPLIFIED DATA")
        print("="*60)
        
        # Remove any rows with all NaN
        clean_df = self.simple_df.dropna(how='all')
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'simplified_volatility_data_{timestamp}.csv'
        
        clean_df.to_csv(filename)
        print(f"Saved to: {filename}")
        print(f"Final shape: {clean_df.shape}")
        
        # Create a summary
        summary = {
            'Date Range': f"{clean_df.index.min()} to {clean_df.index.max()}",
            'Total Days': len(clean_df),
            'Assets': list(clean_df.columns),
            'VIXY Stats': {
                'Mean': clean_df['VIXY_Close'].mean(),
                'Std': clean_df['VIXY_Close'].std(),
                'Min': clean_df['VIXY_Close'].min(),
                'Max': clean_df['VIXY_Close'].max()
            }
        }
        
        print("\nDataset Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return filename, clean_df

def start_from_beginning():
    """Main function to start fresh"""
    
    print("\n" + "="*60)
    print("STARTING FRESH: SIMPLIFYING VIX FUTURES DATA")
    print("="*60)
    
    # Initialize simplifier
    simplifier = VIXFuturesSimplifier()
    
    # Step 1: Analyze current data
    simplifier.analyze_current_data()
    
    # Step 2: Create continuous contract
    simplifier.create_continuous_contract()
    
    # Step 3: Create simplified dataset
    simplifier.create_simple_dataset()
    
    # Step 4: Add VIXY equivalent
    simplifier.add_vixy_equivalent()
    
    # Step 5: Save
    filename, clean_df = simplifier.save_simplified_data()
    
    print("\n" + "="*60)
    print("✅ SIMPLIFICATION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Use '{filename}' for feature engineering")
    print(f"2. Create features using VIXY_Close as the target")
    print(f"3. Train models to predict VIXY movements")
    
    return filename, clean_df

# Run the simplification
if __name__ == "__main__":
    filename, data = start_from_beginning()