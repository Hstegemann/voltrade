import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import zipfile
import os
import json
import hashlib
from datetime import datetime, timedelta
import warnings
import pickle
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ML Volatility Trading Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# EMBEDDED AUTHENTICATION CONFIGURATION
# =====================================================================

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(str.encode(password)).hexdigest()

# User credentials (embedded for Streamlit Cloud compatibility)
FINAL_USERS = {
    "admin": hash_password("VixTrading2025!"),       # Full access admin
    "trader": hash_password("TradingSecure123"),     # Trading access
    "analyst": hash_password("AnalysisView456"),     # View-only access
    "james": hash_password("YourPersonalPass"),      # Your personal account
}

USER_PERMISSIONS = {
    "admin": ["upload", "train", "download", "view", "manage"],
    "trader": ["upload", "train", "view", "download"],
    "analyst": ["view", "download"],
    "james": ["upload", "train", "download", "view", "manage"],  # Your full access
}

PRODUCTION_MODE = False
SHOW_DEMO_CREDENTIALS = not PRODUCTION_MODE

# =====================================================================
# AUTHENTICATION SYSTEM
# =====================================================================

class AuthenticationManager:
    """Authentication system with embedded credentials"""
    
    def __init__(self):
        self.users = FINAL_USERS
        self.permissions = USER_PERMISSIONS
    
    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(str.encode(password)).hexdigest()
    
    def check_password(self, username, password):
        """Check if username/password combination is valid"""
        if username in self.users:
            return self.users[username] == self.hash_password(password)
        return False
    
    def get_user_permissions(self, username):
        """Get permissions for a user"""
        return self.permissions.get(username, [])
    
    def has_permission(self, username, permission):
        """Check if user has specific permission"""
        user_perms = self.get_user_permissions(username)
        return permission in user_perms

def show_login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h1>🔒 ML Volatility Trading Model</h1>
        <h3>Secure Access Required</h3>
        <p>Please login to access the trading model dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("### Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("🔓 Login")
            
            if submitted:
                auth = AuthenticationManager()
                if auth.check_password(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.permissions = auth.get_user_permissions(username)
                    st.session_state.login_time = datetime.now()
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        # Show demo credentials only if enabled in config
        if SHOW_DEMO_CREDENTIALS:
            with st.expander("Demo Credentials", expanded=False):
                st.markdown("""
                **Demo accounts:**
                - **admin** / VixTrading2025! (full access)
                - **trader** / TradingSecure123 (upload & train)
                - **analyst** / AnalysisView456 (view only)
                - **james** / YourPersonalPass (full access)
                
                *⚠️ Change these passwords before production!*
                """)
        
        if PRODUCTION_MODE:
            st.info("🔒 Production mode - Contact administrator for access")

def show_logout_option():
    """Show logout option in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 **Logged in as:** {st.session_state.username}")
    
    # Show user permissions
    perms = st.session_state.get('permissions', [])
    st.sidebar.markdown(f"🔑 **Permissions:** {', '.join(perms)}")
    
    # Show login time
    if 'login_time' in st.session_state:
        login_time = st.session_state.login_time
        st.sidebar.markdown(f"🕐 **Login time:** {login_time.strftime('%H:%M:%S')}")
    
    if st.sidebar.button("🚪 Logout"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def check_permission(permission):
    """Check if user has permission"""
    auth = AuthenticationManager()
    username = st.session_state.get('username', '')
    return auth.has_permission(username, permission)

def require_permission(permission):
    """Show message if user doesn't have permission"""
    if not check_permission(permission):
        st.error(f"❌ Access denied. You need '{permission}' permission for this action.")
        st.info(f"Your permissions: {', '.join(st.session_state.get('permissions', []))}")
        return False
    return True

# =====================================================================
# DATA GENERATION AND ML CLASSES
# =====================================================================

class VIXFuturesSimplifier:
    """Simplified version for demo data generation"""
    
    def __init__(self):
        pass
    
    def create_synthetic_data(self, n_points=10000):
        """Create synthetic VIX and volatility ETF data"""
        np.random.seed(42)
        
        # Generate timestamps (1-minute data)
        start_date = datetime(2024, 1, 1, 9, 30)
        timestamps = [start_date + timedelta(minutes=i) for i in range(n_points)]
        
        # Generate VIX-like data (mean reverting)
        vix_base = 20
        vix_prices = []
        current_vix = vix_base
        
        for _ in range(n_points):
            # Mean reverting random walk
            change = np.random.normal(0, 0.02) - 0.001 * (current_vix - vix_base) / vix_base
            current_vix *= (1 + change)
            current_vix = max(8, min(80, current_vix))  # Keep within reasonable bounds
            vix_prices.append(current_vix)
        
        # Generate VIXY with decay (tracks VIX futures)
        vixy_prices = []
        vixy_start = 20.0
        current_vixy = vixy_start
        daily_decay = -0.0025
        
        for i in range(n_points):
            if i > 0:
                vix_return = (vix_prices[i] - vix_prices[i-1]) / vix_prices[i-1]
                vixy_return = vix_return * 0.5 + daily_decay / 390  # Daily decay spread over 390 minutes
                current_vixy *= (1 + vixy_return)
                current_vixy = max(1, current_vixy)  # Floor at $1
            vixy_prices.append(current_vixy)
        
        # Generate SPX data
        spx_base = 4500
        spx_prices = []
        current_spx = spx_base
        
        for _ in range(n_points):
            change = np.random.normal(0, 0.008)
            current_spx *= (1 + change)
            current_spx = max(3000, min(6000, current_spx))
            spx_prices.append(current_spx)
        
        # Generate HYG and TLT
        hyg_base = 85
        tlt_base = 95
        hyg_prices = [hyg_base * (1 + np.random.normal(0, 0.005)) for _ in range(n_points)]
        tlt_prices = [tlt_base * (1 + np.random.normal(0, 0.003)) for _ in range(n_points)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'VIX_Close': vix_prices,
            'VIXY_Close': vixy_prices,
            'SPX_Close': spx_prices,
            'HYG_Close': hyg_prices,
            'TLT_Close': tlt_prices
        })
        
        df.set_index('Timestamp', inplace=True)
        return df

class FeatureEngineer:
    """Feature engineering for volatility trading"""
    
    def __init__(self, data_df):
        self.df = data_df.copy()
        self.features = pd.DataFrame(index=data_df.index)
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for all assets"""
        
        for col in self.df.columns:
            if '_Close' in col:
                asset_name = col.replace('_Close', '')
                prices = self.df[col]
                
                # Lagged returns
                for lag in [5, 10, 15, 30, 60]:
                    self.features[f'{asset_name}_return_lag{lag}'] = prices.pct_change(lag)
                
                # Moving averages
                self.features[f'{asset_name}_SMA_20'] = prices.rolling(20).mean()
                self.features[f'{asset_name}_SMA_50'] = prices.rolling(50).mean()
                self.features[f'{asset_name}_price_to_SMA20'] = prices / self.features[f'{asset_name}_SMA_20']
                
                # RSI
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                self.features[f'{asset_name}_RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands %B
                sma = prices.rolling(20).mean()
                std = prices.rolling(20).std()
                upper = sma + (2 * std)
                lower = sma - (2 * std)
                self.features[f'{asset_name}_BB_pct'] = (prices - lower) / (upper - lower)
                
                # Volatility
                self.features[f'{asset_name}_volatility_30'] = prices.pct_change().rolling(30).std()
    
    def calculate_cross_asset_features(self):
        """Calculate cross-asset features"""
        
        # VIX/SPX ratio
        if 'VIX_Close' in self.df.columns and 'SPX_Close' in self.df.columns:
            self.features['VIX_SPX_ratio'] = self.df['VIX_Close'] / self.df['SPX_Close'] * 100
            self.features['VIX_SPX_ratio_zscore'] = (
                self.features['VIX_SPX_ratio'] - self.features['VIX_SPX_ratio'].rolling(252).mean()
            ) / self.features['VIX_SPX_ratio'].rolling(252).std()
        
        # HYG/TLT spread
        if 'HYG_Close' in self.df.columns and 'TLT_Close' in self.df.columns:
            self.features['HYG_TLT_ratio'] = self.df['HYG_Close'] / self.df['TLT_Close']
            self.features['HYG_TLT_spread'] = self.df['HYG_Close'] - self.df['TLT_Close']
        
        # VIX regime indicators
        if 'VIX_Close' in self.df.columns:
            self.features['VIX_low_regime'] = (self.df['VIX_Close'] < 15).astype(int)
            self.features['VIX_normal_regime'] = ((self.df['VIX_Close'] >= 15) & (self.df['VIX_Close'] < 25)).astype(int)
            self.features['VIX_high_regime'] = ((self.df['VIX_Close'] >= 25) & (self.df['VIX_Close'] < 35)).astype(int)
            self.features['VIX_extreme_regime'] = (self.df['VIX_Close'] >= 35).astype(int)
    
    def calculate_response_variables(self):
        """Calculate forward-looking response variables"""
        response_df = pd.DataFrame(index=self.df.index)
        
        # Use VIXY as target if available, otherwise VIX
        target_col = 'VIXY_Close' if 'VIXY_Close' in self.df.columns else 'VIX_Close'
        
        if target_col in self.df.columns:
            prices = self.df[target_col]
            
            # Future returns at different horizons
            for horizon in [5, 15, 30, 60]:
                future_price = prices.shift(-horizon)
                
                # Continuous returns
                response_df[f'return_next_{horizon}min'] = (future_price - prices) / prices
                
                # Binary direction
                response_df[f'direction_next_{horizon}min'] = (future_price > prices).astype(int)
        
        return response_df
    
    def engineer_all_features(self):
        """Run all feature engineering"""
        self.calculate_technical_indicators()
        self.calculate_cross_asset_features()
        response_vars = self.calculate_response_variables()
        
        # Combine everything
        all_features = pd.concat([self.df, self.features, response_vars], axis=1)
        
        return all_features

class ModelTrainer:
    """Model training class"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = None
    
    def prepare_data(self, data, target_col='direction_next_15min', test_size=0.2):
        """Prepare data for model training"""
        
        # Identify feature columns
        feature_cols = [col for col in data.columns 
                       if 'next_' not in col  # Not a response variable
                       and col not in ['VIX_Close', 'VIXY_Close', 'SPX_Close', 'HYG_Close', 'TLT_Close']  # Not raw prices
                       and not col.endswith('_Close')]  # Not any close price
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols 
                       if data[col].notna().sum() / len(data) > 0.5]
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Remove rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Handle missing values
        X = X.fillna(method='ffill', limit=5)
        X = X.fillna(0)
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = feature_cols
        
        return len(X), len(feature_cols)
    
    def train_models(self):
        """Train all models"""
        
        results = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(self.X_train_scaled, self.y_train)
        
        y_pred = lr.predict(self.X_test_scaled)
        y_proba = lr.predict_proba(self.X_test_scaled)[:, 1]
        
        results['Logistic Regression'] = {
            'model': lr,
            'predictions': y_pred,
            'probabilities': y_proba,
            'roc_auc': roc_auc_score(self.y_test, y_proba)
        }
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            max_depth=10,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        
        y_pred = rf.predict(self.X_test)
        y_proba = rf.predict_proba(self.X_test)[:, 1]
        
        results['Random Forest'] = {
            'model': rf,
            'predictions': y_pred,
            'probabilities': y_proba,
            'roc_auc': roc_auc_score(self.y_test, y_proba),
            'feature_importance': pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # Try XGBoost if available
        try:
            import xgboost as xgb
            
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            xgb_model.fit(self.X_train, self.y_train, verbose=False)
            
            y_pred = xgb_model.predict(self.X_test)
            y_proba = xgb_model.predict_proba(self.X_test)[:, 1]
            
            results['XGBoost'] = {
                'model': xgb_model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'roc_auc': roc_auc_score(self.y_test, y_proba),
                'feature_importance': pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
            
        except ImportError:
            st.warning("XGBoost not available. Install with: pip install xgboost")
        
        self.results = results
        return results

# =====================================================================
# MAIN STREAMLIT APP
# =====================================================================

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        color: #0d47a1;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Check authentication status
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# If not authenticated, show login page
if not st.session_state.authenticated:
    show_login_page()
    st.stop()  # Stop execution here

# If authenticated, show the main app
st.markdown('<h1 class="main-header">📈 ML Volatility Trading Model</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.selectbox(
    "Choose a page", 
    ["📊 Data Processing", "🔧 Feature Engineering", "🤖 Model Training", "📈 Analysis & Results"]
)

# Show logout option
show_logout_option()

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'engineered_features' not in st.session_state:
    st.session_state.engineered_features = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# =====================================================================
# PAGE CONTENT
# =====================================================================

if page == "📊 Data Processing":
    st.markdown('<h2 class="sub-header">📊 Data Processing</h2>', unsafe_allow_html=True)
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        ["🎲 Generate Demo Data", "📁 Upload Your Files"]
    )
    
    if data_source == "🎲 Generate Demo Data":
        st.markdown("""
        <div class="info-box">
        <h4>Demo Data Generation</h4>
        <p>This will create synthetic market data for testing:</p>
        <ul>
            <li>VIX spot prices (mean-reverting)</li>
            <li>VIXY ETF prices (with realistic decay)</li>
            <li>SPX index prices</li>
            <li>HYG and TLT bond ETF prices</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider("Number of data points", 1000, 50000, 10000, 1000)
        with col2:
            st.metric("Approximate timespan", f"{n_points / 390:.1f} trading days")
        
        if st.button("🎲 Generate Demo Data", type="primary"):
            with st.spinner("Generating synthetic market data..."):
                simplifier = VIXFuturesSimplifier()
                demo_data = simplifier.create_synthetic_data(n_points)
                st.session_state.processed_data = demo_data
                
            st.success(f"✅ Generated {len(demo_data):,} rows of demo data!")
            
            # Show preview
            st.markdown("**Data Preview:**")
            st.dataframe(demo_data.head(10))
            
            # Show summary stats
            st.markdown("**Summary Statistics:**")
            st.dataframe(demo_data.describe())
            
            # Plot the data
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['VIX Level', 'VIXY Price', 'SPX Index', 'HYG vs TLT']
            )
            
            # Sample data for plotting (last 1000 points)
            plot_data = demo_data.tail(1000)
            
            fig.add_trace(
                go.Scatter(x=plot_data.index, y=plot_data['VIX_Close'], name='VIX'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_data.index, y=plot_data['VIXY_Close'], name='VIXY'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=plot_data.index, y=plot_data['SPX_Close'], name='SPX'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_data.index, y=plot_data['HYG_Close'], name='HYG'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=plot_data.index, y=plot_data['TLT_Close'], name='TLT'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    elif data_source == "📁 Upload Your Files":
        # Check upload permission
        if not require_permission("upload"):
            st.stop()
            
        st.markdown("### Upload Your Processed Data")
        
        uploaded_file = st.file_uploader(
            "Upload your merged/processed CSV file", 
            type=['csv'],
            help="Upload output from your VIX futures processor or multi-asset merger"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                st.session_state.processed_data = data
                
                st.success(f"✅ Uploaded {len(data):,} rows of data!")
                
                # Show preview
                st.markdown("**Data Preview:**")
                st.dataframe(data.head(10))
                
                # Show column info
                st.markdown("**Available Columns:**")
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Null %': (data.isnull().sum() / len(data) * 100).round(1)
                })
                st.dataframe(col_info)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

elif page == "🔧 Feature Engineering":
    st.markdown('<h2 class="sub-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first on the Data Processing page.")
    else:
        data = st.session_state.processed_data
        
        st.markdown("### Feature Engineering Configuration")
        
        # Configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Technical Indicators:**")
            include_returns = st.checkbox("Lagged Returns", value=True)
            include_moving_avg = st.checkbox("Moving Averages", value=True)
            include_rsi = st.checkbox("RSI", value=True)
            include_bollinger = st.checkbox("Bollinger Bands", value=True)
        
        with col2:
            st.markdown("**Cross-Asset Features:**")
            include_ratios = st.checkbox("Asset Ratios", value=True)
            include_spreads = st.checkbox("Spreads", value=True)
            include_regimes = st.checkbox("Regime Indicators", value=True)
        
        # Response variable configuration
        st.markdown("**Response Variables:**")
        target_horizons = st.multiselect(
            "Prediction horizons (minutes):",
            [5, 15, 30, 60],
            default=[15, 30]
        )
        
        if st.button("🔧 Engineer Features", type="primary"):
            with st.spinner("Engineering features..."):
                # Initialize feature engineer
                fe = FeatureEngineer(data)
                
                # Run feature engineering
                engineered_data = fe.engineer_all_features()
                
                # Store in session state
                st.session_state.engineered_features = engineered_data
            
            st.success("✅ Feature engineering completed!")
            
            # Show feature summary
            feature_cols = [col for col in engineered_data.columns 
                          if col not in data.columns]
            response_cols = [col for col in engineered_data.columns 
                           if 'next_' in col]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Columns", len(data.columns))
            with col2:
                st.metric("New Features", len(feature_cols))
            with col3:
                st.metric("Response Variables", len(response_cols))
            
            # Show feature categories
            st.markdown("**Feature Categories:**")
            categories = {
                'Returns': len([c for c in feature_cols if 'return' in c]),
                'Technical Indicators': len([c for c in feature_cols if any(x in c for x in ['RSI', 'SMA', 'BB'])]),
                'Cross-Asset': len([c for c in feature_cols if any(x in c for x in ['ratio', 'spread'])]),
                'Regime': len([c for c in feature_cols if 'regime' in c]),
            }
            
            cat_df = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
            st.dataframe(cat_df, use_container_width=True)
            
            # Response variable analysis
            if response_cols:
                st.markdown("**Response Variable Analysis:**")
                
                for resp_col in response_cols[:3]:  # Show first 3
                    if 'direction' in resp_col:
                        valid_data = engineered_data[resp_col].dropna()
                        if len(valid_data) > 0:
                            up_pct = valid_data.mean() * 100
                            st.write(f"**{resp_col}**: {up_pct:.1f}% up moves, {100-up_pct:.1f}% down moves")

elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    # Check training permission
    if not require_permission("train"):
        st.stop()
    
    if st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first.")
    else:
        data = st.session_state.engineered_features
        
        st.markdown("### Model Training Configuration")
        
        # Target selection
        response_cols = [col for col in data.columns if 'direction_next_' in col]
        
        if not response_cols:
            st.error("❌ No response variables found. Please run feature engineering first.")
        else:
            target_col = st.selectbox("Select target variable:", response_cols)
            
            # Training parameters
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
                
            with col2:
                models_to_train = st.multiselect(
                    "Models to train:",
                    ["Logistic Regression", "Random Forest", "XGBoost"],
                    default=["Logistic Regression", "Random Forest"]
                )
            
            if st.button("🤖 Train Models", type="primary"):
                with st.spinner("Training models..."):
                    # Initialize trainer
                    trainer = ModelTrainer()
                    
                    # Prepare data
                    n_samples, n_features = trainer.prepare_data(data, target_col, test_size)
                    
                    # Check class balance
                    train_balance = trainer.y_train.value_counts()
                    test_balance = trainer.y_test.value_counts()
                    
                    st.markdown("**Data Preparation Summary:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", n_samples)
                    with col2:
                        st.metric("Features", n_features)
                    with col3:
                        st.metric("Train/Test Split", f"{len(trainer.X_train)}/{len(trainer.X_test)}")
                    
                    # Show class distribution
                    st.markdown("**Class Distribution:**")
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        st.write("**Training Set:**")
                        st.write(f"Up: {train_balance.get(1, 0):,} ({train_balance.get(1, 0)/len(trainer.y_train)*100:.1f}%)")
                        st.write(f"Down: {train_balance.get(0, 0):,} ({train_balance.get(0, 0)/len(trainer.y_train)*100:.1f}%)")
                    
                    with dist_col2:
                        st.write("**Test Set:**")
                        st.write(f"Up: {test_balance.get(1, 0):,} ({test_balance.get(1, 0)/len(trainer.y_test)*100:.1f}%)")
                        st.write(f"Down: {test_balance.get(0, 0):,} ({test_balance.get(0, 0)/len(trainer.y_test)*100:.1f}%)")
                    
                    # Check if we have both classes
                    if len(train_balance) < 2:
                        st.error("❌ Training set has only one class! This usually means:")
                        st.write("- The target variable is not properly constructed")
                        st.write("- Try a different time horizon")
                        st.write("- Check the forward-looking calculation")
                        
                        # Show diagnostic info
                        target_data = data[target_col].dropna()
                        st.write(f"Target variable `{target_col}` distribution:")
                        st.write(target_data.value_counts())
                        
                    else:
                        # Train models
                        results = trainer.train_models()
                        
                        # Store results
                        st.session_state.model_results = {
                            'trainer': trainer,
                            'results': results,
                            'target_col': target_col
                        }
                        
                        st.success("✅ Model training completed!")
                        
                        # Show results summary
                        st.markdown("**Model Performance Summary:**")
                        
                        perf_data = []
                        for model_name, result in results.items():
                            perf_data.append({
                                'Model': model_name,
                                'ROC-AUC': f"{result['roc_auc']:.4f}",
                                'Accuracy': f"{(result['predictions'] == trainer.y_test).mean():.4f}"
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        # Best model
                        best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
                        best_auc = results[best_model]['roc_auc']
                        
                        st.markdown(f"**🏆 Best Model**: {best_model} (ROC-AUC: {best_auc:.4f})")

elif page == "📈 Analysis & Results":
    st.markdown('<h2 class="sub-header">📈 Analysis & Results</h2>', unsafe_allow_html=True)
    
    # Check view permission
    if not require_permission("view"):
        st.stop()
    
    if st.session_state.model_results is None:
        st.warning("⚠️ Please train models first.")
    else:
        model_data = st.session_state.model_results
        trainer = model_data['trainer']
        results = model_data['results']
        target_col = model_data['target_col']
        
        st.markdown("### Model Performance Analysis")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_auc = results[best_model]['roc_auc']
        n_models = len(results)
        
        with col1:
            st.metric("Best ROC-AUC", f"{best_auc:.4f}")
        with col2:
            st.metric("Models Trained", n_models)
        with col3:
            st.metric("Best Model", best_model)
        
        # Detailed results
        tabs = st.tabs(["📊 Performance Comparison", "🎯 Feature Importance", "📈 Predictions"])
        
        with tabs[0]:
            st.markdown("#### ROC Curves")
            
            # Create ROC curve plot
            fig = go.Figure()
            
            for model_name, result in results.items():
                fpr, tpr, _ = roc_curve(trainer.y_test, result['probabilities'])
                auc = result['roc_auc']
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc:.3f})',
                    line=dict(width=2)
                ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random (AUC = 0.500)',
                showlegend=True
            ))
            
            fig.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=700,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.markdown("#### Feature Importance")
            
            # Show feature importance for models that have it
            for model_name, result in results.items():
                if 'feature_importance' in result:
                    st.markdown(f"**{model_name}**")
                    
                    importance_df = result['feature_importance'].head(15)
                    
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f'Top 15 Features - {model_name}'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    st.dataframe(importance_df.head(10), use_container_width=True)
        
        with tabs[2]:
            st.markdown("#### Prediction Analysis")
            
            # Prediction distributions
            fig = go.Figure()
            
            for model_name, result in results.items():
                fig.add_trace(go.Histogram(
                    x=result['probabilities'],
                    name=model_name,
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Decision Threshold")
            
            fig.update_layout(
                title='Prediction Probability Distributions',
                xaxis_title='Predicted Probability',
                yaxis_title='Count',
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔒 Secure ML Volatility Trading Model | User: {st.session_state.username}</p>
    <p>Session Time: {st.session_state.get('login_time', 'Unknown')}</p>
</div>
""", unsafe_allow_html=True)