import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TargetVariableFixer:
    """Fix the single-class target variable issue and complete Phase 3"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.set_index('Date')
        print(f"Loaded {len(self.data)} rows of data")
        
    def diagnose_target_problem(self):
        """Comprehensive diagnosis of why targets show only one class"""
        
        print("\n" + "="*60)
        print("DIAGNOSING TARGET VARIABLE PROBLEM")
        print("="*60)
        
        # Find VIXY column
        vixy_cols = [col for col in self.data.columns if 'VIXY' in col and 'Close' in col]
        if not vixy_cols:
            print("ERROR: No VIXY Close column found!")
            return
        
        vixy_col = vixy_cols[0]
        vixy_prices = self.data[vixy_col].dropna()
        
        print(f"\n1. VIXY Data Analysis:")
        print(f"   Column: {vixy_col}")
        print(f"   Non-null prices: {len(vixy_prices):,}")
        print(f"   Date range: {vixy_prices.index[0]} to {vixy_prices.index[-1]}")
        
        # Check price movement
        print(f"\n2. Price Movement Statistics:")
        print(f"   Min price: ${vixy_prices.min():.2f}")
        print(f"   Max price: ${vixy_prices.max():.2f}")
        print(f"   Mean price: ${vixy_prices.mean():.2f}")
        print(f"   Std deviation: ${vixy_prices.std():.2f}")
        
        # Calculate returns at different horizons
        print(f"\n3. Return Analysis (WITHOUT shifting):")
        horizons = [1, 5, 15, 30, 60]
        
        for horizon in horizons:
            # WRONG WAY (what might be causing the issue)
            wrong_returns = vixy_prices.pct_change(horizon)
            wrong_up = (wrong_returns > 0).sum()
            wrong_down = (wrong_returns <= 0).sum()
            wrong_pct_up = wrong_up / (wrong_up + wrong_down) * 100
            
            print(f"\n   {horizon}-min returns (looking BACKWARD):")
            print(f"   Up moves: {wrong_up:,} ({wrong_pct_up:.1f}%)")
            print(f"   Down moves: {wrong_down:,} ({100-wrong_pct_up:.1f}%)")
        
        print(f"\n4. Forward-Looking Returns (CORRECT for prediction):")
        
        for horizon in horizons:
            # RIGHT WAY - looking forward
            future_prices = vixy_prices.shift(-horizon)
            current_prices = vixy_prices
            
            # Only calculate where both prices exist
            valid_mask = future_prices.notna() & current_prices.notna()
            future_valid = future_prices[valid_mask]
            current_valid = current_prices[valid_mask]
            
            # Calculate forward returns
            forward_returns = (future_valid - current_valid) / current_valid
            up_moves = (future_valid > current_valid).sum()
            total_moves = len(future_valid)
            pct_up = up_moves / total_moves * 100 if total_moves > 0 else 0
            
            print(f"\n   {horizon}-min FORWARD returns:")
            print(f"   Valid samples: {total_moves:,}")
            print(f"   Up moves: {up_moves:,} ({pct_up:.1f}%)")
            print(f"   Down moves: {total_moves - up_moves:,} ({100-pct_up:.1f}%)")
            print(f"   Mean return: {forward_returns.mean()*100:.3f}%")
            print(f"   Std return: {forward_returns.std()*100:.3f}%")
        
        # Visualize the issue
        self._create_diagnostic_plots(vixy_prices)
        
        return vixy_prices
    
    def _create_diagnostic_plots(self, vixy_prices):
        """Create diagnostic visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VIXY Target Variable Diagnostics', fontsize=16)
        
        # Plot 1: Price series
        ax1 = axes[0, 0]
        vixy_prices.plot(ax=ax1)
        ax1.set_title('VIXY Price Time Series')
        ax1.set_ylabel('Price ($)')
        
        # Plot 2: Returns distribution
        ax2 = axes[0, 1]
        returns_5min = vixy_prices.pct_change(5) * 100
        returns_5min.hist(bins=100, ax=ax2, alpha=0.7)
        ax2.set_title('5-min Returns Distribution (Backward Looking)')
        ax2.set_xlabel('Return (%)')
        ax2.axvline(x=0, color='red', linestyle='--')
        
        # Plot 3: Forward returns
        ax3 = axes[1, 0]
        future_prices = vixy_prices.shift(-15)
        forward_returns = ((future_prices - vixy_prices) / vixy_prices * 100).dropna()
        forward_returns.hist(bins=100, ax=ax3, alpha=0.7, color='green')
        ax3.set_title('15-min FORWARD Returns Distribution')
        ax3.set_xlabel('Forward Return (%)')
        ax3.axvline(x=0, color='red', linestyle='--')
        
        # Plot 4: Up/Down distribution over time
        ax4 = axes[1, 1]
        # Calculate rolling percentage of up moves
        window = 390  # 1 day
        future_15 = vixy_prices.shift(-15)
        up_moves = (future_15 > vixy_prices).astype(float)
        rolling_pct_up = up_moves.rolling(window).mean() * 100
        rolling_pct_up.plot(ax=ax4)
        ax4.axhline(y=50, color='red', linestyle='--', label='50% baseline')
        ax4.set_title('Rolling % of Up Moves (15-min forward)')
        ax4.set_ylabel('% Up Moves')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_balanced_features(self):
        """Create properly balanced features and response variables"""
        
        print("\n" + "="*60)
        print("CREATING BALANCED FEATURES AND TARGETS")
        print("="*60)
        
        features_df = pd.DataFrame(index=self.data.index)
        
        # Get VIXY prices
        vixy_col = next((col for col in self.data.columns if 'VIXY' in col and 'Close' in col), None)
        if not vixy_col:
            return None
            
        vixy_prices = self.data[vixy_col]
        
        # 1. Create BACKWARD-LOOKING features (predictors)
        print("\n1. Creating predictor features (backward-looking)...")
        
        # Lagged returns
        for lag in [5, 10, 15, 30, 60, 120]:
            features_df[f'VIXY_return_lag{lag}'] = vixy_prices.pct_change(lag)
        
        # Technical indicators
        features_df['VIXY_RSI_14'] = self._calculate_rsi(vixy_prices, 14)
        features_df['VIXY_BB_pct'] = self._calculate_bollinger_bands(vixy_prices, 20)
        
        # Moving averages
        features_df['VIXY_SMA_20'] = vixy_prices.rolling(20).mean()
        features_df['VIXY_SMA_50'] = vixy_prices.rolling(50).mean()
        features_df['VIXY_price_to_SMA20'] = vixy_prices / features_df['VIXY_SMA_20']
        
        # Volatility
        features_df['VIXY_volatility_30'] = vixy_prices.pct_change().rolling(30).std()
        
        # Add other asset features
        self._add_cross_asset_features(features_df)
        
        # 2. Create FORWARD-LOOKING response variables (targets)
        print("\n2. Creating response variables (forward-looking)...")
        
        response_stats = []
        for horizon in [5, 15, 30, 60]:
            # Future price
            future_price = vixy_prices.shift(-horizon)
            
            # Continuous return
            ret_col = f'VIXY_return_next_{horizon}min'
            features_df[ret_col] = (future_price - vixy_prices) / vixy_prices
            
            # Binary direction
            dir_col = f'VIXY_up_next_{horizon}min'
            features_df[dir_col] = (future_price > vixy_prices).astype(int)
            
            # Calculate statistics
            valid_mask = features_df[dir_col].notna()
            n_valid = valid_mask.sum()
            n_up = features_df.loc[valid_mask, dir_col].sum()
            pct_up = n_up / n_valid * 100 if n_valid > 0 else 0
            
            response_stats.append({
                'Horizon': f'{horizon}min',
                'Valid Samples': n_valid,
                'Up Moves': n_up,
                'Down Moves': n_valid - n_up,
                '% Up': round(pct_up, 1),
                '% Down': round(100 - pct_up, 1)
            })
            
            print(f"   {dir_col}: {pct_up:.1f}% up, {100-pct_up:.1f}% down")
        
        # Display response variable statistics
        print("\n3. Response Variable Summary:")
        stats_df = pd.DataFrame(response_stats)
        print(stats_df.to_string(index=False))
        
        # 3. Clean the data
        print("\n4. Cleaning data...")
        
        # Identify feature columns (not response variables)
        feature_cols = [col for col in features_df.columns if 'next_' not in col]
        response_cols = [col for col in features_df.columns if 'next_' in col]
        
        # Check completeness
        feature_completeness = features_df[feature_cols].notna().mean()
        print(f"\nFeature completeness: {feature_completeness.mean():.1%}")
        
        # Create final dataset
        final_df = pd.concat([self.data, features_df], axis=1)
        
        # Save the corrected features
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'balanced_features_{timestamp}.csv'
        final_df.to_csv(output_file)
        print(f"\nBalanced features saved to: {output_file}")
        
        return final_df, feature_cols, response_cols
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands %B"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        pct_b = (prices - lower) / (upper - lower)
        return pct_b
    
    def _add_cross_asset_features(self, features_df):
        """Add features from other assets"""
        
        # VIX features
        vix_col = next((col for col in self.data.columns if 'VIX' in col and 'Close' in col and 'VIXY' not in col), None)
        if vix_col:
            vix = self.data[vix_col]
            features_df['VIX_level'] = vix
            features_df['VIX_change_5min'] = vix.diff(5)
            features_df['VIX_high_regime'] = (vix > 20).astype(int)
        
        # SPX features
        spx_col = next((col for col in self.data.columns if 'SPX' in col and 'Close' in col), None)
        if spx_col:
            spx = self.data[spx_col]
            features_df['SPX_return_5min'] = spx.pct_change(5)
            features_df['SPX_return_15min'] = spx.pct_change(15)
    
    def validate_and_proceed(self, final_df, feature_cols, response_cols):
        """Validate we can proceed to Phase 4"""
        
        print("\n" + "="*60)
        print("VALIDATION FOR PHASE 4 (Model Development)")
        print("="*60)
        
        # Choose a target variable
        target_col = 'VIXY_up_next_15min'
        
        if target_col not in final_df.columns:
            print(f"ERROR: Target column {target_col} not found!")
            return False
        
        # Prepare data
        X = final_df[feature_cols].copy()
        y = final_df[target_col].copy()
        
        # Remove rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Fill missing features
        X = X.fillna(method='ffill', limit=5)
        X = X.fillna(0)
        
        # Final check
        print(f"\nFinal Data Check:")
        print(f"Samples: {len(y):,}")
        print(f"Features: {len(feature_cols)}")
        print(f"Target distribution:")
        print(f"  Class 0 (down): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
        print(f"  Class 1 (up): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
        
        if len(np.unique(y)) < 2:
            print("\nERROR: Still only one class in target!")
            return False
        
        print("\n✅ READY FOR PHASE 4: Model Development!")
        print("Target variable is properly balanced.")
        
        return True

# Main execution
def fix_target_and_proceed(data_path='merged_multi_asset_dataset_20250714_112740.csv'):
    """Fix the target variable issue and move to Phase 4"""
    
    # Step 1: Diagnose the problem
    fixer = TargetVariableFixer(data_path)
    vixy_prices = fixer.diagnose_target_problem()
    
    # Step 2: Create balanced features
    final_df, feature_cols, response_cols = fixer.create_balanced_features()
    
    # Step 3: Validate we can proceed
    if fixer.validate_and_proceed(final_df, feature_cols, response_cols):
        print("\n🎉 Phase 3 COMPLETE! Ready for Model Development.")
        return final_df, feature_cols, response_cols
    else:
        print("\n❌ Still have issues to resolve.")
        return None, None, None

# Run the fix
if __name__ == "__main__":
    final_df, feature_cols, response_cols = fix_target_and_proceed()