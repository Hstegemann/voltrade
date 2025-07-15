"""
ML VOLATILITY TRADING - COMPLETE NEXT STEPS IMPLEMENTATION
=========================================================

This script provides a step-by-step implementation for:
1. Running feature engineering on your merged dataset
2. Training initial ML models
3. Evaluating results and generating predictions
4. Creating a simple trading strategy

Author: ML Volatility Trading Project
Date: January 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ML VOLATILITY TRADING - NEXT STEPS GUIDE")
print("="*80)
print("\nThis script will walk you through each step sequentially.")
print("Make sure you have your merged dataset file ready!")
print("")

# =====================================================================
# STEP 1: LOAD AND VALIDATE YOUR MERGED DATA
# =====================================================================

def step1_load_merged_data():
    """Step 1: Load and validate the merged dataset"""
    print("\n" + "="*60)
    print("STEP 1: LOADING YOUR MERGED DATASET")
    print("="*60)
    
    # Get file path
    print("\nPlease enter the path to your merged dataset CSV file")
    print("(or press Enter to use the most recent merged file in current directory)")
    
    file_path = input("\nFile path: ").strip()
    
    if not file_path:
        # Try to find the most recent merged file
        import glob
        import os
        
        merged_files = glob.glob("merged_*.csv")
        if merged_files:
            # Get most recent file
            file_path = max(merged_files, key=os.path.getctime)
            print(f"\nUsing most recent file: {file_path}")
        else:
            print("No merged files found in current directory!")
            return None
    
    # Load the data
    print(f"\nLoading {file_path}...")
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded successfully: {len(df):,} rows × {len(df.columns)} columns")
        
        # Show basic info
        print("\nDataset Info:")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Show available columns by asset
        print("\nAvailable columns by asset:")
        assets = {}
        for col in df.columns:
            asset = col.split('_')[0]
            if asset not in assets:
                assets[asset] = []
            assets[asset].append(col)
        
        for asset, cols in assets.items():
            print(f"  {asset}: {len(cols)} columns")
        
        return df
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# =====================================================================
# STEP 2: ENHANCED FEATURE ENGINEERING
# =====================================================================

def calculate_cross_asset_features(df):
    """Calculate cross-asset features like spreads and ratios"""
    print("\nCalculating cross-asset features...")
    features = pd.DataFrame(index=df.index)
    
    # 1. VIX/SPX ratio (fear gauge)
    if 'VIX_Close' in df.columns and 'SPX_Close' in df.columns:
        features['VIX_SPX_ratio'] = df['VIX_Close'] / df['SPX_Close'] * 100
        features['VIX_SPX_ratio_zscore'] = (
            features['VIX_SPX_ratio'] - features['VIX_SPX_ratio'].rolling(252).mean()
        ) / features['VIX_SPX_ratio'].rolling(252).std()
    
    # 2. HYG/TLT spread (risk on/off indicator)
    if 'HYG_Close' in df.columns and 'TLT_Close' in df.columns:
        features['HYG_TLT_ratio'] = df['HYG_Close'] / df['TLT_Close']
        features['HYG_TLT_spread'] = df['HYG_Close'] - df['TLT_Close']
        
        # Rate of change
        features['HYG_TLT_ratio_change'] = features['HYG_TLT_ratio'].pct_change(20)
    
    # 3. Term structure slope (if you have futures data)
    if 'VIX_Front_Month_Close' in df.columns and 'VIX_Next_Month_Close' in df.columns:
        features['VIX_term_slope'] = (
            df['VIX_Next_Month_Close'] - df['VIX_Front_Month_Close']
        )
        features['VIX_contango'] = (features['VIX_term_slope'] > 0).astype(int)
        
        # Rolling percentile of term structure
        features['VIX_slope_percentile'] = features['VIX_term_slope'].rolling(1000).rank(pct=True)
    
    # 4. Cross-asset momentum
    if 'SPX_Close' in df.columns and 'DXY_Close' in df.columns:
        # Dollar vs Stocks
        features['SPX_DXY_ratio'] = df['SPX_Close'] / df['DXY_Close']
        features['SPX_DXY_momentum'] = features['SPX_DXY_ratio'].pct_change(60)
    
    # 5. Volatility regime indicators
    if 'VIX_Close' in df.columns:
        # VIX regime levels
        features['VIX_low_regime'] = (df['VIX_Close'] < 15).astype(int)
        features['VIX_normal_regime'] = ((df['VIX_Close'] >= 15) & (df['VIX_Close'] < 25)).astype(int)
        features['VIX_high_regime'] = ((df['VIX_Close'] >= 25) & (df['VIX_Close'] < 35)).astype(int)
        features['VIX_extreme_regime'] = (df['VIX_Close'] >= 35).astype(int)
    
    print(f"✓ Created {len(features.columns)} cross-asset features")
    return features

def step2_feature_engineering(df):
    """Step 2: Complete feature engineering"""
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Try to import from feature_engineer.py
    try:
        from feature_engineer import FeatureEngineer
        print("✓ Using FeatureEngineer from feature_engineer.py")
    except ImportError:
        print("⚠️ Could not import from feature_engineer.py")
        print("  Using embedded FeatureEngineer instead...")
        
        # Embedded minimal FeatureEngineer class
        class FeatureEngineer:
            def __init__(self, data_df):
                self.df = data_df.copy()
                self.features = pd.DataFrame(index=data_df.index)
            
            def calculate_all_features(self):
                """Calculate basic features"""
                print("Calculating basic features...")
                
                # Simple returns for main assets
                for col in self.df.columns:
                    if 'Close' in col and 'Unnamed' not in col:
                        # Log returns
                        for period in [5, 10, 30, 60]:
                            self.features[f'{col}_return_{period}m'] = np.log(
                                self.df[col] / self.df[col].shift(period)
                            )
                        
                        # Simple moving averages
                        self.features[f'{col}_SMA_20'] = self.df[col].rolling(20).mean()
                        self.features[f'{col}_SMA_50'] = self.df[col].rolling(50).mean()
                        
                        # RSI simplified
                        delta = self.df[col].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        self.features[f'{col}_RSI'] = 100 - (100 / (1 + rs))
                
                # VIX specific features if available
                if 'VIX_Close' in self.df.columns:
                    self.features['VIX_level'] = self.df['VIX_Close']
                    self.features['VIX_high'] = (self.df['VIX_Close'] > 20).astype(int)
                    self.features['VIX_extreme'] = (self.df['VIX_Close'] > 30).astype(int)
                
                return self.features
            
            def calculate_response_variables(self):
                """Calculate response variables"""
                response_df = pd.DataFrame(index=self.df.index)
                
                # Find a suitable price column for response
                price_col = None
                for col in ['VIX_Continuous_30D', 'VIX_Close', 'VIXY_Close']:
                    if col in self.df.columns:
                        price_col = col
                        break
                
                if price_col:
                    # Future returns
                    for minutes in [5, 10, 15, 30, 60]:
                        response_df[f'return_next_{minutes}min'] = np.log(
                            self.df[price_col].shift(-minutes) / self.df[price_col]
                        )
                        response_df[f'direction_next_{minutes}min'] = (
                            response_df[f'return_next_{minutes}min'] > 0
                        ).astype(int)
                
                return response_df
    
    # Initialize feature engineer
    fe = FeatureEngineer(df)
    
    # Calculate all base features
    print("\nCalculating base features...")
    features = fe.calculate_all_features()
    
    # Add cross-asset features
    cross_features = calculate_cross_asset_features(df)
    features = pd.concat([features, cross_features], axis=1)
    
    # Calculate response variables
    print("\nCalculating response variables...")
    response_vars = fe.calculate_response_variables()
    
    # Combine everything
    ml_dataset = pd.concat([df, features, response_vars], axis=1)
    
    # Save the feature-engineered dataset
    output_file = f"ml_dataset_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ml_dataset.to_csv(output_file)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"✓ Total features: {len(features.columns)}")
    print(f"✓ Total response variables: {len(response_vars.columns)}")
    print(f"✓ Saved to: {output_file}")
    
    # Show feature categories
    print("\nFeature Summary:")
    feature_categories = {
        'Price-based': len([c for c in features.columns if any(x in c for x in ['return', 'price', 'Close'])]),
        'Technical': len([c for c in features.columns if any(x in c for x in ['RSI', 'MACD', 'BB', 'SMA'])]),
        'Volume': len([c for c in features.columns if 'volume' in c.lower()]),
        'Cross-asset': len([c for c in features.columns if any(x in c for x in ['ratio', 'spread', 'HYG_TLT', 'SPX_DXY'])]),
        'Regime': len([c for c in features.columns if 'regime' in c]),
        'Term Structure': len([c for c in features.columns if any(x in c for x in ['term', 'slope', 'contango'])])
    }
    
    for category, count in feature_categories.items():
        print(f"  {category}: {count} features")
    
    return ml_dataset

# =====================================================================
# STEP 3: TRAIN AND EVALUATE MODELS
# =====================================================================

def step3_train_models(ml_dataset):
    """Step 3: Train initial ML models"""
    print("\n" + "="*60)
    print("STEP 3: TRAINING ML MODELS")
    print("="*60)
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    
    # Try to import xgboost, but don't fail if it's not installed
    try:
        import xgboost as xgb
        has_xgboost = True
    except ImportError:
        print("⚠️ XGBoost not installed. Install with: pip install xgboost")
        print("   Continuing without XGBoost...")
        has_xgboost = False
    
    # Select target variable
    print("\nAvailable target variables:")
    targets = [col for col in ml_dataset.columns if 'direction_next' in col]
    
    if not targets:
        print("⚠️ No target variables found! Looking for alternatives...")
        # Try to find return columns
        return_targets = [col for col in ml_dataset.columns if 'return_next' in col]
        if return_targets:
            print(f"Found {len(return_targets)} return targets. Creating binary targets...")
            # Create binary targets from returns
            for ret_col in return_targets:
                direction_col = ret_col.replace('return_next', 'direction_next')
                ml_dataset[direction_col] = (ml_dataset[ret_col] > 0).astype(int)
                targets.append(direction_col)
        else:
            print("\n❌ No response variables found in dataset!")
            print("Available columns that might work:")
            close_cols = [col for col in ml_dataset.columns if 'Close' in col and 'Unnamed' not in col]
            for col in close_cols[:10]:
                print(f"  - {col}")
            print("\nPlease check that response variables were calculated correctly.")
            return None, None, None, None, None
    
    for i, target in enumerate(targets, 1):
        print(f"  {i}. {target}")
    
    if not targets:
        print("\n❌ No valid target variables available!")
        return None, None, None, None, None
    
    target_choice = input(f"\nSelect target variable (1-{len(targets)}, default=1): ").strip()
    
    if target_choice and target_choice.isdigit() and 1 <= int(target_choice) <= len(targets):
        target_col = targets[int(target_choice) - 1]
    else:
        target_col = targets[0]
    
    print(f"\nUsing target: {target_col}")
    
    # Select features
    exclude_patterns = ['return_next', 'direction_next', 'category_next', 
                       '_Open', '_High', '_Low', '_Close', '_Volume',
                       'Front_Month', 'Next_Month', 'Continuous', 'Weight']
    
    feature_cols = [col for col in ml_dataset.columns if 
                   not any(pattern in col for pattern in exclude_patterns) and
                   col != target_col]
    
    # Clean data
    print(f"\nPreparing data...")
    print(f"  Initial features: {len(feature_cols)}")
    
    # Remove features with too many NaN values
    nan_threshold = 0.5  # Remove features with >50% NaN
    valid_features = []
    for col in feature_cols:
        if ml_dataset[col].notna().sum() / len(ml_dataset) > nan_threshold:
            valid_features.append(col)
    
    feature_cols = valid_features
    print(f"  Features after NaN filter: {len(feature_cols)}")
    
    # Create clean dataset
    clean_data = ml_dataset[feature_cols + [target_col]].dropna()
    print(f"  Samples after cleaning: {len(clean_data):,}")
    
    # Check target distribution
    print(f"\nTarget variable distribution:")
    print(clean_data[target_col].value_counts())
    print(f"Percentage positive: {clean_data[target_col].mean()*100:.1f}%")
    
    if clean_data[target_col].nunique() < 2:
        print("\n❌ ERROR: Target variable has only one class!")
        print("This usually means:")
        print("  1. The price only moved in one direction during this period")
        print("  2. The response calculation might be incorrect")
        print("  3. Try a different target timeframe (5min, 15min, etc.)")
        
        # Show some diagnostics
        if 'return_next_30min' in ml_dataset.columns:
            returns = ml_dataset['return_next_30min'].dropna()
            print(f"\nReturn statistics:")
            print(f"  Mean return: {returns.mean():.4f}")
            print(f"  Positive returns: {(returns > 0).sum()} / {len(returns)}")
            print(f"  Negative returns: {(returns < 0).sum()} / {len(returns)}")
            print(f"  Zero returns: {(returns == 0).sum()} / {len(returns)}")
        
        return None, None, None, None, None
    
    if len(clean_data) < 1000:
        print("\n⚠️ Warning: Very few samples after cleaning. Consider:")
        print("  - Using a different target variable")
        print("  - Adjusting NaN threshold")
        print("  - Using imputation instead of dropping")
    
    X = clean_data[feature_cols]
    y = clean_data[target_col]
    
    # Time series split
    n_splits = 3  # Reduced for faster training
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    # Add XGBoost if available
    if has_xgboost:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    else:
        print("\n  (Skipping XGBoost model)")
    
    results = {}
    
    # Train each model
    for model_name, model in models.items():
        print(f"\n" + "-"*50)
        print(f"Training {model_name}...")
        
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            print(f"  Fold {fold}/{n_splits}...", end='')
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            if model_name == 'XGBoost':
                model.fit(X_train_scaled, y_train, verbose=False)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            scores.append(score)
            
            print(f" ROC-AUC: {score:.4f}")
        
        avg_score = np.mean(scores)
        print(f"\n{model_name} Average ROC-AUC: {avg_score:.4f}")
        
        results[model_name] = {
            'scores': scores,
            'avg_score': avg_score,
            'model': model,
            'scaler': scaler
        }
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['avg_score'])
    print(f"\n✓ Best model: {best_model} (ROC-AUC: {results[best_model]['avg_score']:.4f})")
    
    # Feature importance for best model
    if hasattr(results[best_model]['model'], 'feature_importances_'):
        print(f"\nTop 15 Important Features ({best_model}):")
        importances = results[best_model]['model'].feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_imp_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return results, X, y, feature_cols, target_col

# =====================================================================
# STEP 4: GENERATE TRADING SIGNALS
# =====================================================================

def step4_generate_signals(ml_dataset, results, feature_cols, target_col):
    """Step 4: Generate trading signals"""
    print("\n" + "="*60)
    print("STEP 4: GENERATING TRADING SIGNALS")
    print("="*60)
    
    # Use best model
    best_model_name = max(results, key=lambda x: results[x]['avg_score'])
    best_model = results[best_model_name]['model']
    scaler = results[best_model_name]['scaler']
    
    print(f"Using {best_model_name} for signal generation")
    
    # Prepare data for prediction
    prediction_data = ml_dataset[feature_cols].copy()
    
    # Forward fill missing values for prediction
    prediction_data = prediction_data.fillna(method='ffill', limit=5)
    
    # Generate predictions for all available data
    print("\nGenerating predictions...")
    predictions = []
    valid_indices = []
    
    # Process in chunks to handle missing data
    chunk_size = 1000
    for i in range(0, len(prediction_data), chunk_size):
        chunk = prediction_data.iloc[i:i+chunk_size]
        
        # Skip rows with too many NaN values
        nan_counts = chunk.isnull().sum(axis=1)
        valid_mask = nan_counts < (len(feature_cols) * 0.3)  # Less than 30% NaN
        
        if valid_mask.sum() > 0:
            valid_chunk = chunk[valid_mask]
            valid_indices.extend(valid_chunk.index)
            
            # Scale and predict
            chunk_scaled = scaler.transform(valid_chunk)
            chunk_preds = best_model.predict_proba(chunk_scaled)[:, 1]
            predictions.extend(chunk_preds)
    
    # Create signals DataFrame
    signals_df = pd.DataFrame(index=valid_indices)
    signals_df['prediction_probability'] = predictions
    signals_df['signal'] = (signals_df['prediction_probability'] > 0.5).astype(int)
    
    # Add signal strength
    signals_df['signal_strength'] = pd.cut(
        signals_df['prediction_probability'],
        bins=[0, 0.3, 0.45, 0.55, 0.7, 1.0],
        labels=['Strong_Short', 'Short', 'Neutral', 'Long', 'Strong_Long']
    )
    
    # Calculate signal statistics
    print(f"\nSignal Statistics:")
    print(f"  Total signals generated: {len(signals_df):,}")
    print(f"  Long signals: {signals_df['signal'].sum():,} ({signals_df['signal'].sum()/len(signals_df)*100:.1f}%)")
    print(f"  Short signals: {(1-signals_df['signal']).sum():,} ({(1-signals_df['signal']).sum()/len(signals_df)*100:.1f}%)")
    
    print("\nSignal Strength Distribution:")
    print(signals_df['signal_strength'].value_counts().sort_index())
    
    # Save signals
    output_file = f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    signals_df.to_csv(output_file)
    print(f"\n✓ Signals saved to: {output_file}")
    
    return signals_df

# =====================================================================
# STEP 5: SIMPLE BACKTEST
# =====================================================================

def step5_backtest(ml_dataset, signals_df):
    """Step 5: Simple backtesting of signals"""
    print("\n" + "="*60)
    print("STEP 5: BACKTESTING SIGNALS")
    print("="*60)
    
    # Merge signals with price data
    if 'VIX_Continuous_30D' in ml_dataset.columns:
        price_col = 'VIX_Continuous_30D'
    elif 'VIX_Close' in ml_dataset.columns:
        price_col = 'VIX_Close'
    else:
        print("No suitable price column found for backtesting!")
        return None
    
    backtest_df = signals_df.copy()
    backtest_df['price'] = ml_dataset.loc[signals_df.index, price_col]
    
    # Calculate returns
    backtest_df['price_return'] = backtest_df['price'].pct_change()
    
    # Strategy returns (long when signal=1)
    backtest_df['strategy_return'] = backtest_df['signal'].shift(1) * backtest_df['price_return']
    
    # Calculate cumulative returns
    backtest_df['cumulative_return'] = (1 + backtest_df['strategy_return']).cumprod() - 1
    backtest_df['buy_hold_return'] = (1 + backtest_df['price_return']).cumprod() - 1
    
    # Performance metrics
    total_return = backtest_df['cumulative_return'].iloc[-1]
    buy_hold_return = backtest_df['buy_hold_return'].iloc[-1]
    
    # Win rate
    winning_trades = backtest_df[backtest_df['strategy_return'] > 0]['strategy_return'].count()
    total_trades = backtest_df[backtest_df['strategy_return'] != 0]['strategy_return'].count()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Sharpe ratio (simplified)
    strategy_sharpe = backtest_df['strategy_return'].mean() / backtest_df['strategy_return'].std() * np.sqrt(252 * 390)
    
    print(f"\nBacktest Results:")
    print(f"  Strategy Return: {total_return*100:.2f}%")
    print(f"  Buy & Hold Return: {buy_hold_return*100:.2f}%")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"  Total Trades: {total_trades:,}")
    
    # Save backtest results
    output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    backtest_df.to_csv(output_file)
    print(f"\n✓ Backtest results saved to: {output_file}")
    
    return backtest_df

# =====================================================================
# MAIN EXECUTION FLOW
# =====================================================================

def main():
    """Main execution flow"""
    print("\nLet's walk through each step of your ML pipeline.\n")
    
    # Step 1: Load data
    df = step1_load_merged_data()
    if df is None:
        print("\nCannot proceed without data. Exiting.")
        return
    
    input("\nPress Enter to continue to Feature Engineering...")
    
    # Step 2: Feature engineering
    try:
        ml_dataset = step2_feature_engineering(df)
    except Exception as e:
        print(f"\nError in feature engineering: {e}")
        print("Make sure you have the ml_volatility_pipeline.py file in the same directory!")
        return
    
    input("\nPress Enter to continue to Model Training...")
    
    # Step 3: Train models
    try:
        results, X, y, feature_cols, target_col = step3_train_models(ml_dataset)
    except Exception as e:
        print(f"\nError in model training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    input("\nPress Enter to continue to Signal Generation...")
    
    # Step 4: Generate signals
    try:
        signals_df = step4_generate_signals(ml_dataset, results, feature_cols, target_col)
    except Exception as e:
        print(f"\nError in signal generation: {e}")
        return
    
    input("\nPress Enter to continue to Backtesting...")
    
    # Step 5: Backtest
    try:
        backtest_df = step5_backtest(ml_dataset, signals_df)
    except Exception as e:
        print(f"\nError in backtesting: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nYou have successfully:")
    print("✓ Engineered features from your multi-asset dataset")
    print("✓ Trained and evaluated 3 different ML models")
    print("✓ Generated trading signals")
    print("✓ Backtested your strategy")
    print("\nNext steps:")
    print("1. Analyze the feature importance results")
    print("2. Try different target variables (5min, 15min, 1hr)")
    print("3. Experiment with different probability thresholds")
    print("4. Add transaction costs to backtesting")
    print("5. Implement position sizing and risk management")

if __name__ == "__main__":
    main()