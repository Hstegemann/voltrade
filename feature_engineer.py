"""
Feature Engineering Module for ML Volatility Trading
Save this file as: feature_engineer.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for volatility trading ML model
    Based on ML_Model_Project_Outline specifications
    """
    
    def __init__(self, data_df):
        self.df = data_df.copy()
        self.features = pd.DataFrame(index=data_df.index)
        
    def calculate_log_returns(self, price_series, periods):
        """Calculate log returns for specified periods"""
        returns = {}
        for period in periods:
            returns[f'return_{period}'] = np.log(price_series / price_series.shift(period))
        return pd.DataFrame(returns)
    
    def calculate_lagged_returns(self):
        """Calculate lagged returns as specified in the outline"""
        print("Calculating lagged returns...")
        
        # Define lag periods (in minutes for 1-minute bars)
        lag_periods = {
            '5min': 5,
            '10min': 10,
            '15min': 15,
            '30min': 30,
            '1hr': 60,
            '2hr': 120,
            '1day': 390,  # Regular trading hours
            '1week': 1950  # 5 trading days
        }
        
        # Calculate for each asset
        for col in self.df.columns:
            if 'Close' in col and 'Unnamed' not in col:
                asset_name = col.replace('_Close', '')
                for lag_name, lag_minutes in lag_periods.items():
                    feature_name = f'{asset_name}_lag_return_{lag_name}'
                    self.features[feature_name] = np.log(
                        self.df[col] / self.df[col].shift(lag_minutes)
                    )
                    
        return self.features
    
    def calculate_macd(self, price_series, fast=12, slow=26, signal=9):
        """Calculate MACD indicators"""
        # Convert periods to minutes (assuming 1-minute bars)
        fast_minutes = fast * 30  # Approximate for intraday
        slow_minutes = slow * 30
        signal_minutes = signal * 30
        
        # Calculate EMAs
        ema_fast = price_series.ewm(span=fast_minutes, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow_minutes, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal_minutes, adjust=False).mean()
        
        # MACD histogram
        macd_histogram = macd_line - signal_line
        
        return macd_line, signal_line, macd_histogram
    
    def calculate_rsi(self, price_series, period=14):
        """Calculate RSI"""
        # Convert to minutes for intraday
        period_minutes = period * 30  # Approximate
        
        # Calculate price changes
        delta = price_series.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses (Wilder's smoothing)
        avg_gains = gains.ewm(alpha=1/period_minutes, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period_minutes, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_term_structure_features(self):
        """Calculate futures term structure features"""
        print("Calculating term structure features...")
        
        # Slope between front and next month
        if 'VIX_Front_Month_Close' in self.df.columns and 'VIX_Next_Month_Close' in self.df.columns:
            # Simple slope
            self.features['VIX_term_slope'] = (
                self.df['VIX_Next_Month_Close'] - self.df['VIX_Front_Month_Close']
            ) / 30  # Approximate days between contracts
            
            # Contango/Backwardation flag
            self.features['VIX_contango'] = (
                self.df['VIX_Next_Month_Close'] > self.df['VIX_Front_Month_Close']
            ).astype(int)
            
            # Calculate z-score of slope
            lookback_periods = [60, 120, 240]  # 1hr, 2hr, 4hr
            for period in lookback_periods:
                mean_slope = self.features['VIX_term_slope'].rolling(period).mean()
                std_slope = self.features['VIX_term_slope'].rolling(period).std()
                self.features[f'VIX_slope_zscore_{period}'] = (
                    (self.features['VIX_term_slope'] - mean_slope) / std_slope
                )
                
        return self.features
    
    def calculate_volume_spike(self):
        """Calculate volume spike features"""
        print("Calculating volume spikes...")
        
        # Look for volume columns
        volume_cols = [col for col in self.df.columns if 'Volume' in col and 'Unnamed' not in col]
        
        if volume_cols:
            lookback_map = {
                5: 30,    # 5 min -> 30 bar lookback
                10: 60,   # 10 min -> 60 bar lookback
                15: 90,   # 15 min -> 90 bar lookback
                30: 150,  # 30 min -> 150 bar lookback
                60: 210,  # 1 hr -> 210 bar lookback
                120: 300, # 2 hr -> 300 bar lookback
                240: 540  # 4 hr -> 540 bar lookback
            }
            
            for vol_col in volume_cols:
                asset_name = vol_col.replace('_Volume', '')
                for aggregate_mins, lookback_bars in lookback_map.items():
                    # Rolling mean and std
                    vol_mean = self.df[vol_col].rolling(lookback_bars).mean()
                    vol_std = self.df[vol_col].rolling(lookback_bars).std()
                    
                    # Z-score (volume spike)
                    self.features[f'{asset_name}_volume_spike_{aggregate_mins}min'] = (
                        (self.df[vol_col] - vol_mean) / vol_std
                    )
                
        return self.features
    
    def calculate_bollinger_bands(self, price_series, period=20, num_std=2):
        """Calculate Bollinger Bands %B"""
        # Middle band (SMA)
        middle_band = price_series.rolling(period).mean()
        
        # Standard deviation
        std_dev = price_series.rolling(period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        # %B calculation
        percent_b = (price_series - lower_band) / (upper_band - lower_band)
        
        return percent_b, upper_band, lower_band
    
    def calculate_vix_features(self):
        """Calculate VIX-specific features"""
        print("Calculating VIX features...")
        
        if 'VIX_Close' in self.df.columns:
            # VIX absolute level
            self.features['VIX_level'] = self.df['VIX_Close']
            
            # VIX changes
            for minutes in [5, 10, 15, 30, 60]:
                self.features[f'VIX_change_{minutes}min'] = (
                    self.df['VIX_Close'] - self.df['VIX_Close'].shift(minutes)
                )
            
            # VIX percentile over last N days
            for days in [5, 10, 20, 30]:
                bars = days * 390  # Trading hours per day
                self.features[f'VIX_percentile_{days}d'] = (
                    self.df['VIX_Close'].rolling(bars).rank(pct=True)
                )
                
        return self.features
    
    def calculate_spy_features(self):
        """Calculate SPY/SPX features"""
        print("Calculating SPX features...")
        
        if 'SPX_Close' in self.df.columns:
            # Returns
            for minutes in [1, 5, 10, 30, 60]:
                self.features[f'SPX_return_{minutes}min'] = (
                    np.log(self.df['SPX_Close'] / self.df['SPX_Close'].shift(minutes))
                )
            
            # Intraday volatility (rolling std)
            for window in [30, 60, 120]:
                returns = self.df['SPX_Close'].pct_change()
                self.features[f'SPX_volatility_{window}min'] = (
                    returns.rolling(window).std() * np.sqrt(252 * 390)  # Annualized
                )
                
        return self.features
    
    def calculate_all_features(self):
        """Calculate all features specified in the project outline"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING - ML Model Project")
        print("="*60)
        
        # 1. Lagged returns
        self.calculate_lagged_returns()
        
        # 2. VIX features
        self.calculate_vix_features()
        
        # 3. SPY/SPX features  
        self.calculate_spy_features()
        
        # 4. Term structure features
        self.calculate_term_structure_features()
        
        # 5. Volume features (if available)
        self.calculate_volume_spike()
        
        # 6. Technical indicators
        print("Calculating technical indicators...")
        
        # MACD for main assets
        for col in self.df.columns:
            if 'Close' in col and 'Unnamed' not in col:
                asset_name = col.replace('_Close', '')
                macd, signal, hist = self.calculate_macd(self.df[col])
                self.features[f'{asset_name}_MACD'] = macd
                self.features[f'{asset_name}_MACD_signal'] = signal
                self.features[f'{asset_name}_MACD_histogram'] = hist
        
        # RSI for main assets
        for col in self.df.columns:
            if 'Close' in col and 'Unnamed' not in col:
                asset_name = col.replace('_Close', '')
                self.features[f'{asset_name}_RSI'] = self.calculate_rsi(self.df[col])
        
        # Bollinger Bands %B
        for col in self.df.columns:
            if 'Close' in col and 'Unnamed' not in col and any(x in col for x in ['VIX', 'SPX']):
                asset_name = col.replace('_Close', '')
                bb_pct, _, _ = self.calculate_bollinger_bands(self.df[col])
                self.features[f'{asset_name}_BB_pct'] = bb_pct
        
        # 7. Regime indicators
        print("Calculating regime indicators...")
        if 'VIX_Close' in self.df.columns:
            self.features['regime_high_vix'] = (self.df['VIX_Close'] > 25).astype(int)
            self.features['regime_extreme_vix'] = (self.df['VIX_Close'] > 30).astype(int)
        
        print(f"\n✓ Total features calculated: {len(self.features.columns)}")
        return self.features
    
    def calculate_response_variables(self):
        """Calculate response variables for model training"""
        print("\nCalculating response variables...")
        
        response_df = pd.DataFrame(index=self.df.index)
        
        # Find best column for response - prioritize continuous contract
        response_col = None
        candidates = ['VIX_Continuous_30D', 'VIX_Close', 'VIXY_Close', 'VIX_Front_Month_Close']
        
        print("Looking for response column...")
        for col in candidates:
            if col in self.df.columns:
                response_col = col
                print(f"✓ Found: {col}")
                break
            else:
                print(f"  - {col}: Not found")
        
        # If still not found, try any Close column
        if not response_col:
            print("  Trying any Close column...")
            close_cols = [col for col in self.df.columns if 'Close' in col and 'Unnamed' not in col]
            if close_cols:
                response_col = close_cols[0]
                print(f"✓ Using: {response_col}")
        
        if response_col:
            print(f"Using {response_col} for response variables")
            
            lead_periods = {
                '5min': 5,
                '10min': 10,
                '15min': 15,
                '30min': 30,
                '1hr': 60,
                '2hr': 120
            }
            
            for period_name, minutes in lead_periods.items():
                # Continuous returns - FIXED: ensure proper calculation
                future_price = self.df[response_col].shift(-minutes)
                current_price = self.df[response_col]
                
                # Calculate returns
                response_df[f'return_next_{period_name}'] = (
                    (future_price - current_price) / current_price
                )
                
                # Binary classification (up/down)
                response_df[f'direction_next_{period_name}'] = (
                    future_price > current_price
                ).astype(int)
                
                # Debug first calculation
                if period_name == '30min':
                    print(f"\nDebug {period_name} calculation:")
                    valid_mask = response_df[f'return_next_{period_name}'].notna()
                    valid_returns = response_df.loc[valid_mask, f'return_next_{period_name}']
                    valid_directions = response_df.loc[valid_mask, f'direction_next_{period_name}']
                    
                    if len(valid_returns) > 0:
                        print(f"  Valid samples: {len(valid_returns):,}")
                        print(f"  Mean return: {valid_returns.mean():.4%}")
                        print(f"  Positive: {valid_directions.sum():,} ({valid_directions.mean():.1%})")
                        print(f"  Negative: {(1-valid_directions).sum():,} ({(1-valid_directions.mean()):.1%})")
                        
                        # Show sample of actual vs future prices
                        sample_idx = valid_returns.index[:5]
                        print("\n  Sample calculations:")
                        for idx in sample_idx:
                            curr = current_price.loc[idx]
                            fut = future_price.loc[idx] if idx in future_price.index else np.nan
                            if not pd.isna(fut):
                                print(f"    {idx}: {curr:.2f} -> {fut:.2f} = {(fut-curr)/curr:.2%}")
                    else:
                        print("  ERROR: No valid returns calculated!")
            
            # Categorical classification for key timeframes
            for period_name in ['30min', '1hr']:
                return_col = f'return_next_{period_name}'
                if return_col in response_df.columns:
                    # Calculate percentiles for thresholds
                    valid_returns = response_df[return_col].dropna()
                    if len(valid_returns) > 100:
                        thresholds = valid_returns.quantile([0.1, 0.3, 0.7, 0.9])
                        
                        response_df[f'category_next_{period_name}'] = pd.cut(
                            response_df[return_col],
                            bins=[-np.inf] + thresholds.tolist() + [np.inf],
                            labels=['Very_Weak', 'Weak', 'Uncertain', 'Strong', 'Very_Strong']
                        )
        else:
            print("❌ ERROR: No suitable column found for response variables!")
            print("Available columns:")
            for i, col in enumerate(self.df.columns[:20]):
                print(f"  {i+1}. {col}")
            if len(self.df.columns) > 20:
                print(f"  ... and {len(self.df.columns) - 20} more")
        
        print(f"✓ Response variables calculated: {len(response_df.columns)}")
        return response_df

# Test function
if __name__ == "__main__":
    print("FeatureEngineer class loaded successfully!")
    print("Available methods:")
    for method in dir(FeatureEngineer):
        if not method.startswith('_'):
            print(f"  - {method}")