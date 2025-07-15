import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json

class VolatilityModelDevelopment:
    """Phase 4: Complete Model Development Pipeline"""
    
    def __init__(self, data_path):
        """Initialize with feature data"""
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self, target_col='VIXY_up_next_15min', test_size=0.2):
        """Prepare data for model training"""
        
        print("\n" + "="*60)
        print("PREPARING DATA FOR MODEL TRAINING")
        print("="*60)
        
        # Identify features and target
        feature_cols = [col for col in self.data.columns 
                       if 'next_' not in col  # Not a response variable
                       and col not in ['Open', 'High', 'Low', 'Close', 'Volume']  # Not raw price data
                       and 'Date' not in col]  # Not date column
        
        # Remove any remaining price columns
        feature_cols = [col for col in feature_cols 
                       if not any(x in col for x in ['_Open', '_High', '_Low', '_Close']) 
                       or any(x in col for x in ['return', 'RSI', 'MACD', 'BB', 'ratio', 'volatility'])]
        
        print(f"\nSelected {len(feature_cols)} features")
        print(f"Target variable: {target_col}")
        
        # Extract features and target
        X = self.data[feature_cols].copy()
        y = self.data[target_col].copy()
        
        # Remove rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\nValid samples: {len(y):,}")
        
        # Handle missing values
        print("\nHandling missing values...")
        # Forward fill first (for time series continuity)
        X = X.fillna(method='ffill', limit=5)
        # Then backward fill
        X = X.fillna(method='bfill', limit=5)
        # Fill remaining with 0
        X = X.fillna(0)
        
        # Check for infinite values
        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.sum() > 0:
            print(f"Removing {inf_mask.sum()} rows with infinite values")
            X = X[~inf_mask]
            y = y[~inf_mask]
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"\nTrain set: {len(self.X_train):,} samples")
        print(f"Test set: {len(self.X_test):,} samples")
        
        # Check class distribution
        print(f"\nTrain set distribution:")
        print(f"  Up: {(self.y_train == 1).sum():,} ({(self.y_train == 1).mean()*100:.1f}%)")
        print(f"  Down: {(self.y_train == 0).sum():,} ({(self.y_train == 0).mean()*100:.1f}%)")
        
        print(f"\nTest set distribution:")
        print(f"  Up: {(self.y_test == 1).sum():,} ({(self.y_test == 1).mean()*100:.1f}%)")
        print(f"  Down: {(self.y_test == 0).sum():,} ({(self.y_test == 0).mean()*100:.1f}%)")
        
        self.feature_names = feature_cols
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return True
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        
        print("\n" + "-"*40)
        print("TRAINING LOGISTIC REGRESSION")
        print("-"*40)
        
        # Train model
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        
        lr.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = lr.predict(self.X_test_scaled)
        y_proba = lr.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        self._evaluate_model('Logistic Regression', lr, y_pred, y_proba)
        
        # Store model
        self.models['logistic_regression'] = lr
        
        return lr
    
    def train_random_forest(self):
        """Train Random Forest model"""
        
        print("\n" + "-"*40)
        print("TRAINING RANDOM FOREST")
        print("-"*40)
        
        # Train model
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=10  # Prevent overfitting
        )
        
        rf.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = rf.predict(self.X_test)
        y_proba = rf.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        self._evaluate_model('Random Forest', rf, y_pred, y_proba)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Important Features:")
        print(importance_df.head(15).to_string(index=False))
        
        self.feature_importance['random_forest'] = importance_df
        self.models['random_forest'] = rf
        
        return rf
    
    def train_xgboost(self):
        """Train XGBoost model"""
        
        print("\n" + "-"*40)
        print("TRAINING XGBOOST")
        print("-"*40)
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Make predictions
        y_pred = xgb_model.predict(self.X_test)
        y_proba = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        self._evaluate_model('XGBoost', xgb_model, y_pred, y_proba)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost'] = importance_df
        self.models['xgboost'] = xgb_model
        
        return xgb_model
    
    def _evaluate_model(self, model_name, model, y_pred, y_proba):
        """Evaluate model performance"""
        
        print(f"\n{model_name} Results:")
        print(classification_report(self.y_test, y_pred))
        
        # ROC-AUC
        roc_auc = roc_auc_score(self.y_test, y_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Store results
        self.results[model_name] = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'roc_auc': roc_auc,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
    
    def cross_validate_models(self):
        """Perform time series cross-validation"""
        
        print("\n" + "="*60)
        print("TIME SERIES CROSS-VALIDATION")
        print("="*60)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nCross-validating {model_name}...")
            
            scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
                # Get fold data
                X_fold_train = self.X_train.iloc[train_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_val = self.y_train.iloc[val_idx]
                
                # Scale if needed
                if model_name == 'Logistic Regression':
                    scaler = StandardScaler()
                    X_fold_train = scaler.fit_transform(X_fold_train)
                    X_fold_val = scaler.transform(X_fold_val)
                
                # Train and evaluate
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                
                y_proba = model_copy.predict_proba(X_fold_val)[:, 1]
                fold_auc = roc_auc_score(y_fold_val, y_proba)
                scores.append(fold_auc)
                
                print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")
            
            cv_results[model_name] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            
            print(f"  Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        self.cv_results = cv_results
        
        return cv_results
    
    def create_visualizations(self):
        """Create model performance visualizations"""
        
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
            auc = results['roc_auc']
            ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Importance Comparison
        ax2 = axes[0, 1]
        if 'random_forest' in self.feature_importance:
            top_features = self.feature_importance['random_forest'].head(10)
            ax2.barh(top_features['feature'], top_features['importance'])
            ax2.set_xlabel('Importance')
            ax2.set_title('Top 10 Features (Random Forest)')
            ax2.invert_yaxis()
        
        # 3. Confusion Matrix
        ax3 = axes[1, 0]
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['roc_auc'])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title(f'Confusion Matrix ({best_model_name})')
        
        # 4. Prediction Distribution
        ax4 = axes[1, 1]
        for model_name, results in self.results.items():
            ax4.hist(results['probabilities'], bins=50, alpha=0.5, label=model_name)
        
        ax4.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Distributions')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models_and_results(self):
        """Save trained models and results"""
        
        print("\n" + "="*60)
        print("SAVING MODELS AND RESULTS")
        print("="*60)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        for model_name, model in self.models.items():
            filename = f'model_{model_name}_{timestamp}.pkl'
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, f'scaler_{timestamp}.pkl')
        
        # Save results
        results_summary = {
            'timestamp': timestamp,
            'models': {},
            'cross_validation': self.cv_results if hasattr(self, 'cv_results') else None,
            'feature_names': self.feature_names
        }
        
        for model_name, results in self.results.items():
            results_summary['models'][model_name] = {
                'roc_auc': results['roc_auc'],
                'classification_report': results['classification_report']
            }
        
        with open(f'model_results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        # Save feature importance
        for model_name, importance_df in self.feature_importance.items():
            importance_df.to_csv(f'feature_importance_{model_name}_{timestamp}.csv', index=False)
        
        print("\nAll models and results saved successfully!")
        
        return timestamp
    
    def get_best_model(self):
        """Identify and return the best performing model"""
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['roc_auc'])
        best_auc = self.results[best_model_name]['roc_auc']
        
        print(f"\n🏆 Best Model: {best_model_name} with ROC-AUC = {best_auc:.4f}")
        
        return self.models[best_model_name], best_model_name

def run_complete_model_development(data_path):
    """Run the complete Phase 4 pipeline"""
    
    print("\n" + "="*60)
    print("PHASE 4: MODEL DEVELOPMENT")
    print("="*60)
    
    # Initialize
    model_dev = VolatilityModelDevelopment(data_path)
    
    # Prepare data
    if not model_dev.prepare_data():
        print("ERROR: Failed to prepare data!")
        return None
    
    # Train models
    model_dev.train_logistic_regression()
    model_dev.train_random_forest()
    model_dev.train_xgboost()
    
    # Cross-validate
    model_dev.cross_validate_models()
    
    # Create visualizations
    model_dev.create_visualizations()
    
    # Save everything
    timestamp = model_dev.save_models_and_results()
    
    # Get best model
    best_model, best_name = model_dev.get_best_model()
    
    print("\n" + "="*60)
    print("✅ PHASE 4 COMPLETE!")
    print("="*60)
    print(f"\nNext: Phase 5 - Model Evaluation and Backtesting")
    print(f"Best model ({best_name}) saved with timestamp: {timestamp}")
    
    return model_dev

# Run Phase 4
if __name__ == "__main__":
    # Use the balanced features file from Phase 3
    model_dev = run_complete_model_development('balanced_features_XXXXXX.csv')