import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
from datetime import datetime
import pickle
import zipfile
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="ML Volatility Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataManager:
    """
    Comprehensive data management system for the ML volatility platform
    """
    
    def __init__(self):
        self.data_dir = "data"
        self.shared_data_dir = os.path.join(self.data_dir, "shared")
        self.user_data_dir = os.path.join(self.data_dir, "users")
        self.metadata_file = os.path.join(self.data_dir, "metadata.json")
        
        # Create directories
        for directory in [self.data_dir, self.shared_data_dir, self.user_data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize metadata
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load dataset metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"shared_datasets": {}, "user_datasets": {}}
    
    def save_metadata(self):
        """Save dataset metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def upload_shared_dataset(self, file, name, description, uploader):
        """Upload a dataset to be shared with all users"""
        try:
            # Generate unique filename
            file_hash = hashlib.md5(file.getvalue()).hexdigest()[:8]
            filename = f"{name}_{file_hash}.csv"
            filepath = os.path.join(self.shared_data_dir, filename)
            
            # Read and validate data
            df = pd.read_csv(file)
            
            # Save file
            df.to_csv(filepath, index=False)
            
            # Update metadata
            self.metadata["shared_datasets"][filename] = {
                "name": name,
                "description": description,
                "uploader": uploader,
                "upload_date": datetime.now().isoformat(),
                "shape": df.shape,
                "columns": list(df.columns),
                "file_size_mb": os.path.getsize(filepath) / (1024 * 1024)
            }
            
            self.save_metadata()
            return True, f"Dataset '{name}' uploaded successfully!"
            
        except Exception as e:
            return False, f"Error uploading dataset: {str(e)}"
    
    def upload_user_dataset(self, file, name, username):
        """Upload a dataset for a specific user"""
        try:
            # Create user directory
            user_dir = os.path.join(self.user_data_dir, username)
            os.makedirs(user_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.csv"
            filepath = os.path.join(user_dir, filename)
            
            # Read and save data
            df = pd.read_csv(file)
            df.to_csv(filepath, index=False)
            
            # Update metadata
            if username not in self.metadata["user_datasets"]:
                self.metadata["user_datasets"][username] = {}
            
            self.metadata["user_datasets"][username][filename] = {
                "name": name,
                "upload_date": datetime.now().isoformat(),
                "shape": df.shape,
                "columns": list(df.columns),
                "file_size_mb": os.path.getsize(filepath) / (1024 * 1024)
            }
            
            self.save_metadata()
            return True, f"Personal dataset '{name}' uploaded successfully!"
            
        except Exception as e:
            return False, f"Error uploading dataset: {str(e)}"
    
    def get_shared_datasets(self):
        """Get list of shared datasets"""
        return self.metadata.get("shared_datasets", {})
    
    def get_user_datasets(self, username):
        """Get list of user's personal datasets"""
        return self.metadata.get("user_datasets", {}).get(username, {})
    
    def load_shared_dataset(self, filename):
        """Load a shared dataset"""
        filepath = os.path.join(self.shared_data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None
    
    def load_user_dataset(self, username, filename):
        """Load a user's personal dataset"""
        filepath = os.path.join(self.user_data_dir, username, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None
    
    def delete_dataset(self, dataset_type, username, filename):
        """Delete a dataset"""
        try:
            if dataset_type == "shared":
                filepath = os.path.join(self.shared_data_dir, filename)
                if filename in self.metadata["shared_datasets"]:
                    del self.metadata["shared_datasets"][filename]
            else:
                filepath = os.path.join(self.user_data_dir, username, filename)
                if username in self.metadata["user_datasets"] and filename in self.metadata["user_datasets"][username]:
                    del self.metadata["user_datasets"][username][filename]
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            self.save_metadata()
            return True, "Dataset deleted successfully!"
            
        except Exception as e:
            return False, f"Error deleting dataset: {str(e)}"

# Authentication System
def check_password():
    """Simple authentication system"""
    def password_entered():
        users = {
            "admin": "admin123",
            "trader": "trade456", 
            "analyst": "analyze789",
            "james": "james2025"
        }
        
        username = st.session_state["username"].lower()
        password = st.session_state["password"]
        
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = username
            st.session_state["current_user"] = username
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["authenticated"] = False
            st.error("Invalid username or password")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔐 ML Volatility Trading Platform - Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered)
            
            with st.expander("👤 Demo Accounts"):
                st.write("**Available accounts:**")
                st.write("• admin / admin123 - Full access")
                st.write("• trader / trade456 - Trading focus") 
                st.write("• analyst / analyze789 - Analysis focus")
                st.write("• james / james2025 - Project owner")
        
        return False
    
    return True

def create_sample_datasets():
    """Create sample datasets for demonstration"""
    data_manager = DataManager()
    
    # Sample VIX/SPX volatility data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 09:30:00', periods=10000, freq='1min')
    
    sample_data = {
        'DateTime': dates,
        'VIX_Close': 20 + np.cumsum(np.random.randn(10000) * 0.1),
        'SPX_Close': 4500 + np.cumsum(np.random.randn(10000) * 0.5),
        'VIXY_Close': 15 + np.cumsum(np.random.randn(10000) * 0.08),
        'HYG_Close': 80 + np.cumsum(np.random.randn(10000) * 0.02),
        'TLT_Close': 95 + np.cumsum(np.random.randn(10000) * 0.03),
        'Volume_VIX': np.random.randint(100000, 1000000, 10000),
        'Volume_SPX': np.random.randint(500000, 5000000, 10000)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add realistic correlations
    df['VIX_Close'] = 20 + (4500 - df['SPX_Close']) * 0.01 + np.random.randn(10000) * 2
    df['VIXY_Close'] = df['VIX_Close'] * 0.7 + np.random.randn(10000) * 1
    
    # Save as sample data
    sample_file = BytesIO()
    df.to_csv(sample_file, index=False)
    sample_file.seek(0)
    
    return sample_file, df

def show_progress_tracker():
    """Show project progress in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.header("📋 Project Progress")
        
        progress_data = {
            "✅ Phase 1: Environment Setup": 100,
            "✅ Phase 2: Data Integration": 100, 
            "🔄 Phase 3: Feature Engineering": 85,
            "🔄 Phase 4: Model Development": 60,
            "❌ Phase 5: Model Evaluation": 20,
            "❌ Phase 6: Backtesting": 0,
            "❌ Phase 7: Live Trading": 0
        }
        
        for phase, progress in progress_data.items():
            st.write(phase)
            st.progress(progress / 100)
        
        overall_progress = sum(progress_data.values()) / len(progress_data)
        st.metric("Overall Progress", f"{overall_progress:.0f}%")
        
        st.markdown("**🎯 Next Steps:**")
        st.write("• Fix single-class target issue")
        st.write("• Complete feature engineering")
        st.write("• Train multiple models")
        st.write("• Implement backtesting")

def main():
    """Main application"""
    
    # Authentication check
    if not check_password():
        return
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Main title
    st.title("📈 ML Volatility Trading Platform")
    st.markdown(f"**Welcome, {st.session_state['current_user'].title()}!** | Role: {st.session_state['user_role'].title()}")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🗂️ Data Management")
        
        page = st.selectbox(
            "Choose Section:",
            ["📊 Data Overview", "📤 Upload Data", "📥 Browse Datasets", "🔬 Data Analysis", "🤖 ML Pipeline", "⚙️ Admin Panel"]
        )
        
        # User info
        st.markdown("---")
        st.markdown(f"**Current User:** {st.session_state['current_user']}")
        if st.button("🚪 Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        
        # Progress tracker
        show_progress_tracker()
    
    # Page routing
    if page == "📊 Data Overview":
        show_data_overview(data_manager)
    elif page == "📤 Upload Data":
        show_upload_page(data_manager)
    elif page == "📥 Browse Datasets":
        show_browse_page(data_manager)
    elif page == "🔬 Data Analysis":
        show_analysis_page(data_manager)
    elif page == "🤖 ML Pipeline":
        show_ml_pipeline(data_manager)
    elif page == "⚙️ Admin Panel":
        show_admin_panel(data_manager)

def show_data_overview(data_manager):
    """Show overview of all available datasets"""
    st.header("📊 Data Overview")
    
    # Quick setup if no data exists
    shared_datasets = data_manager.get_shared_datasets()
    user_datasets = data_manager.get_user_datasets(st.session_state['current_user'])
    
    if not shared_datasets and not user_datasets:
        st.info("No datasets found. Let's set up some sample data!")
        
        if st.button("🧪 Create Sample Volatility Data"):
            with st.spinner("Creating sample dataset..."):
                sample_file, sample_df = create_sample_datasets()
                
                success, message = data_manager.upload_shared_dataset(
                    sample_file, "Sample VIX SPX Data", 
                    "Sample volatility dataset with VIX, SPX, VIXY data for testing ML models",
                    st.session_state['current_user']
                )
                
                if success:
                    st.success(message)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)
        return
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Shared Datasets", len(shared_datasets))
    with col2:
        st.metric("Your Datasets", len(user_datasets))
    with col3:
        total_size = sum([d.get('file_size_mb', 0) for d in shared_datasets.values()])
        total_size += sum([d.get('file_size_mb', 0) for d in user_datasets.values()])
        st.metric("Total Data Size", f"{total_size:.1f} MB")
    
    # Recent uploads
    st.subheader("📈 Available Datasets")
    
    tab1, tab2 = st.tabs(["🌐 Shared Datasets", "👤 My Datasets"])
    
    with tab1:
        if shared_datasets:
            for filename, metadata in shared_datasets.items():
                with st.expander(f"📁 {metadata['name']}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Description:** {metadata['description']}")
                        st.write(f"**Uploader:** {metadata['uploader']}")
                        st.write(f"**Shape:** {metadata['shape'][0]:,} rows × {metadata['shape'][1]} columns")
                    with col2:
                        st.write(f"**Size:** {metadata['file_size_mb']:.1f} MB")
                        st.write(f"**Date:** {metadata['upload_date'][:10]}")
                        if st.button(f"📥 Load", key=f"load_shared_{filename}"):
                            st.session_state['selected_dataset'] = ('shared', filename)
                            st.success(f"Dataset '{metadata['name']}' loaded!")
                            st.rerun()
        else:
            st.info("No shared datasets available.")
    
    with tab2:
        if user_datasets:
            for filename, metadata in user_datasets.items():
                with st.expander(f"📁 {metadata['name']}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Upload Date:** {metadata['upload_date'][:10]}")
                        st.write(f"**Shape:** {metadata['shape'][0]:,} rows × {metadata['shape'][1]} columns")
                    with col2:
                        st.write(f"**Size:** {metadata['file_size_mb']:.1f} MB")
                        if st.button(f"📥 Load", key=f"load_user_{filename}"):
                            st.session_state['selected_dataset'] = ('user', filename)
                            st.success(f"Dataset '{metadata['name']}' loaded!")
                            st.rerun()
        else:
            st.info("You haven't uploaded any personal datasets yet.")

def show_upload_page(data_manager):
    """Show data upload interface"""
    st.header("📤 Upload Data")
    
    # Upload type selection
    upload_type = st.radio(
        "Upload Type:",
        ["📊 Personal Dataset", "🌐 Shared Dataset (All Users)"],
        help="Personal datasets are only visible to you. Shared datasets are visible to all users."
    )
    
    is_shared = "Shared" in upload_type
    
    # Only admin can upload shared datasets
    if is_shared and st.session_state['user_role'] != 'admin':
        st.warning("Only administrators can upload shared datasets. You can upload personal datasets.")
        is_shared = False
    
    st.markdown("---")
    
    # Upload form
    with st.form("upload_form"):
        st.subheader("📋 Dataset Information")
        
        dataset_name = st.text_input(
            "Dataset Name *",
            placeholder="e.g., VIX_SPX_Q1_2024",
            help="Give your dataset a descriptive name"
        )
        
        if is_shared:
            description = st.text_area(
                "Description *",
                placeholder="Describe what this dataset contains, time period, features, etc.",
                help="Help other users understand your dataset"
            )
        
        uploaded_file = st.file_uploader(
            "Choose CSV File *",
            type=['csv'],
            help="Upload a CSV file with your volatility trading data"
        )
        
        # Data requirements
        with st.expander("📋 Data Requirements"):
            st.markdown("""
            **Your CSV should include:**
            - Timestamp/DateTime column
            - Price data (VIX, SPX, VIXY, etc.)
            - Volume data (optional)
            - Clean, properly formatted data
            
            **Recommended columns:**
            - DateTime, VIX_Close, SPX_Close, VIXY_Close, HYG_Close, TLT_Close
            """)
        
        submitted = st.form_submit_button("🚀 Upload Dataset")
        
        if submitted:
            if not dataset_name:
                st.error("Please provide a dataset name")
            elif is_shared and not description:
                st.error("Please provide a description for shared datasets")
            elif uploaded_file is None:
                st.error("Please upload a CSV file")
            else:
                # Process upload
                with st.spinner("Processing upload..."):
                    if is_shared:
                        success, message = data_manager.upload_shared_dataset(
                            uploaded_file, dataset_name, description, 
                            st.session_state['current_user']
                        )
                    else:
                        success, message = data_manager.upload_user_dataset(
                            uploaded_file, dataset_name, st.session_state['current_user']
                        )
                    
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
    
    # Quick sample data creation
    st.markdown("---")
    st.subheader("🧪 Create Sample Dataset")
    st.info("Don't have data? Create a sample volatility dataset for testing!")
    
    if st.button("📊 Generate Sample VIX/SPX Data"):
        with st.spinner("Creating sample dataset..."):
            sample_file, sample_df = create_sample_datasets()
            
            # Auto-upload sample data
            success, message = data_manager.upload_user_dataset(
                sample_file, "Sample_VIX_SPX_Data", st.session_state['current_user']
            )
            
            if success:
                st.success("Sample dataset created and uploaded!")
                st.dataframe(sample_df.head())
            else:
                st.error(f"Error creating sample: {message}")

def show_browse_page(data_manager):
    """Show dataset browsing interface"""
    st.header("📥 Browse Datasets")
    
    # Dataset type tabs
    tab1, tab2 = st.tabs(["🌐 Shared Datasets", "👤 My Datasets"])
    
    with tab1:
        st.subheader("🌐 Shared Datasets")
        shared_datasets = data_manager.get_shared_datasets()
        
        if shared_datasets:
            for filename, metadata in shared_datasets.items():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**📊 {metadata['name']}**")
                        st.markdown(f"*{metadata['description']}*")
                        st.markdown(f"👤 By: {metadata['uploader']} | 📅 {metadata['upload_date'][:10]}")
                    
                    with col2:
                        st.markdown(f"**{metadata['shape'][0]:,}** rows")
                        st.markdown(f"**{metadata['shape'][1]}** columns")
                        st.markdown(f"**{metadata['file_size_mb']:.1f}** MB")
                    
                    with col3:
                        if st.button("👁️ Preview", key=f"preview_shared_{filename}"):
                            df = data_manager.load_shared_dataset(filename)
                            if df is not None:
                                st.dataframe(df.head(10))
                        
                        if st.button("📥 Load", key=f"load_shared_{filename}"):
                            st.session_state['selected_dataset'] = ('shared', filename)
                            st.success(f"Loaded: {metadata['name']}")
                    
                    st.markdown("---")
        else:
            st.info("No shared datasets available yet.")
    
    with tab2:
        st.subheader("👤 My Personal Datasets")
        user_datasets = data_manager.get_user_datasets(st.session_state['current_user'])
        
        if user_datasets:
            for filename, metadata in user_datasets.items():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**📊 {metadata['name']}**")
                        st.markdown(f"📅 Uploaded: {metadata['upload_date'][:10]}")
                    
                    with col2:
                        st.markdown(f"**{metadata['shape'][0]:,}** rows")
                        st.markdown(f"**{metadata['shape'][1]}** columns")
                        st.markdown(f"**{metadata['file_size_mb']:.1f}** MB")
                    
                    with col3:
                        if st.button("👁️ Preview", key=f"preview_user_{filename}"):
                            df = data_manager.load_user_dataset(st.session_state['current_user'], filename)
                            if df is not None:
                                st.dataframe(df.head(10))
                        
                        col3a, col3b = st.columns(2)
                        with col3a:
                            if st.button("📥", key=f"load_user_{filename}", help="Load dataset"):
                                st.session_state['selected_dataset'] = ('user', filename)
                                st.success(f"Loaded: {metadata['name']}")
                        
                        with col3b:
                            if st.button("🗑️", key=f"delete_user_{filename}", help="Delete dataset"):
                                success, message = data_manager.delete_dataset('user', st.session_state['current_user'], filename)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    
                    st.markdown("---")
        else:
            st.info("You haven't uploaded any personal datasets yet.")

def show_analysis_page(data_manager):
    """Show data analysis interface"""
    st.header("🔬 Data Analysis")
    
    # Check if dataset is loaded
    if 'selected_dataset' not in st.session_state:
        st.warning("Please load a dataset first from the Browse Datasets page.")
        
        # Show available datasets
        shared_datasets = data_manager.get_shared_datasets()
        if shared_datasets:
            st.subheader("📥 Quick Load")
            for filename, metadata in list(shared_datasets.items())[:3]:
                if st.button(f"Load: {metadata['name']}", key=f"quick_load_{filename}"):
                    st.session_state['selected_dataset'] = ('shared', filename)
                    st.rerun()
        return
    
    dataset_type, filename = st.session_state['selected_dataset']
    
    # Load dataset
    if dataset_type == 'shared':
        df = data_manager.load_shared_dataset(filename)
        metadata = data_manager.get_shared_datasets()[filename]
    else:
        df = data_manager.load_user_dataset(st.session_state['current_user'], filename)
        metadata = data_manager.get_user_datasets(st.session_state['current_user'])[filename]
    
    if df is None:
        st.error("Could not load dataset.")
        return
    
    st.success(f"📊 Analyzing: **{metadata['name']}**")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📊 Visualizations", "🔍 Missing Data", "📈 Correlations"])
    
    with tab1:
        st.subheader("📋 Data Preview")
        
        # Data preview
        st.dataframe(df.head(20))
        
        # Column info
        st.subheader("📊 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null': df.count(),
            'Missing': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info)
    
    with tab2:
        st.subheader("📊 Data Visualizations")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Time series plots
            selected_cols = st.multiselect(
                "Select columns to plot:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    sample_data = df[col].head(1000) if len(df) > 1000 else df[col]
                    fig.add_trace(go.Scatter(
                        y=sample_data,
                        name=col,
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title="Time Series Plot",
                    xaxis_title="Time Index",
                    yaxis_title="Value",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            st.subheader("📊 Distributions")
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            
            fig = px.histogram(df.head(10000), x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for visualization.")
    
    with tab3:
        st.subheader("🔍 Missing Data Analysis")
        
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            }).sort_values('Missing %', ascending=False)
            
            # Filter only columns with missing data
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_df.empty:
                st.dataframe(missing_df)
                
                # Missing data chart
                fig = px.bar(missing_df, x='Column', y='Missing %', 
                           title="Missing Data by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing data found!")
        else:
            st.success("No missing data found!")
    
    with tab4:
        st.subheader("📈 Correlation Analysis")
        
        if len(numeric_cols) > 1:
            # Sample data for large datasets
            sample_df = df.head(5000) if len(df) > 5000 else df
            corr_matrix = sample_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlations
            st.subheader("🔍 High Correlations (>0.7)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(high_corr_df)
            else:
                st.info("No high correlations (>0.7) found.")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

def show_ml_pipeline(data_manager):
    """Show ML pipeline interface"""
    st.header("🤖 ML Pipeline")
    
    # Check if dataset is loaded
    if 'selected_dataset' not in st.session_state:
        st.warning("Please load a dataset first from the Browse Datasets page.")
        return
    
    dataset_type, filename = st.session_state['selected_dataset']
    
    # Load dataset
    if dataset_type == 'shared':
        df = data_manager.load_shared_dataset(filename)
        metadata = data_manager.get_shared_datasets()[filename]
    else:
        df = data_manager.load_user_dataset(st.session_state['current_user'], filename)
        metadata = data_manager.get_user_datasets(st.session_state['current_user'])[filename]
    
    if df is None:
        st.error("Could not load dataset.")
        return
    
    st.success(f"🤖 ML Pipeline for: **{metadata['name']}**")
    
    # Pipeline tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Feature Engineering", "🎯 Target Creation", "🤖 Model Training", "📊 Results"])
    
    with tab1:
        st.subheader("⚙️ Feature Engineering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found for feature engineering.")
            return
        
        st.write("**Available numeric columns:**")
        st.write(numeric_cols)
        
        # Basic features
        if st.button("🔧 Create Basic Features"):
            with st.spinner("Creating features..."):
                feature_df = df.copy()
                
                # Lagged returns for first few numeric columns
                for col in numeric_cols[:3]:
                    if col in feature_df.columns:
                        for lag in [5, 15, 30]:
                            if len(feature_df) > lag:
                                feature_df[f'{col}_return_{lag}'] = np.log(feature_df[col] / feature_df[col].shift(lag))
                
                # Simple moving averages
                for col in numeric_cols[:3]:
                    if col in feature_df.columns:
                        feature_df[f'{col}_MA_20'] = feature_df[col].rolling(20).mean()
                        feature_df[f'{col}_vs_MA'] = feature_df[col] / feature_df[f'{col}_MA_20'] - 1
                
                st.session_state['feature_df'] = feature_df
                st.success(f"Created {feature_df.shape[1] - df.shape[1]} new features!")
                st.dataframe(feature_df.head())
    
    with tab2:
        st.subheader("🎯 Target Variable Creation")
        
        if 'feature_df' not in st.session_state:
            st.warning("Please create features first.")
            return
        
        feature_df = st.session_state['feature_df']
        
        # Select target column
        target_candidates = [col for col in feature_df.columns if any(x in col.lower() for x in ['vix', 'vixy', 'close'])]
        
        if not target_candidates:
            target_candidates = feature_df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        
        target_col = st.selectbox("Select column for target creation:", target_candidates)
        
        prediction_horizon = st.slider("Prediction horizon (periods ahead):", 1, 60, 15)
        
        if st.button("🎯 Create Target Variable"):
            # Create binary target
            future_returns = np.log(feature_df[target_col].shift(-prediction_horizon) / feature_df[target_col])
            binary_target = (future_returns > 0).astype(int)
            
            feature_df[f'target_{prediction_horizon}'] = binary_target
            
            st.session_state['feature_df'] = feature_df
            st.session_state['target_col'] = f'target_{prediction_horizon}'
            
            # Show target distribution
            target_dist = binary_target.value_counts()
            st.write("**Target Distribution:**")
            st.write(target_dist)
            
            if len(target_dist) < 2:
                st.error("⚠️ Target has only one class! Try different prediction horizon or column.")
            else:
                st.success("✅ Binary target created successfully!")
    
    with tab3:
        st.subheader("🤖 Model Training")
        
        if 'feature_df' not in st.session_state or 'target_col' not in st.session_state:
            st.warning("Please create features and target first.")
            return
        
        feature_df = st.session_state['feature_df']
        target_col = st.session_state['target_col']
        
        # Select features for training
        feature_cols = [col for col in feature_df.columns if 'return_' in col or 'MA' in col or 'vs_' in col]
        
        if not feature_cols:
            feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != target_col][:10]
        
        selected_features = st.multiselect(
            "Select features for training:",
            feature_cols,
            default=feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
        )
        
        if st.button("🚀 Train Models") and selected_features:
            with st.spinner("Training models..."):
                try:
                    # Prepare data
                    X = feature_df[selected_features].dropna()
                    y = feature_df.loc[X.index, target_col]
                    
                    # Check for sufficient data and classes
                    if len(X) < 100:
                        st.error("Not enough data points after removing NaN values.")
                        return
                    
                    if len(y.unique()) < 2:
                        st.error("Target variable has only one class.")
                        return
                    
                    # Train/test split (time series aware)
                    split_point = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
                    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    models = {
                        'Logistic Regression': LogisticRegression(random_state=42),
                        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
                    }
                    
                    results = {}
                    
                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        
                        auc = roc_auc_score(y_test, y_pred_proba)
                        
                        results[name] = {
                            'model': model,
                            'auc': auc,
                            'predictions': y_pred,
                            'probabilities': y_pred_proba
                        }
                    
                    st.session_state['ml_results'] = results
                    st.session_state['test_data'] = (X_test, y_test)
                    st.session_state['scaler'] = scaler
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Show results
                    col1, col2 = st.columns(2)
                    for i, (name, result) in enumerate(results.items()):
                        with [col1, col2][i]:
                            st.metric(f"{name} AUC", f"{result['auc']:.3f}")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    with tab4:
        st.subheader("📊 Model Results")
        
        if 'ml_results' not in st.session_state:
            st.warning("Please train models first.")
            return
        
        results = st.session_state['ml_results']
        X_test, y_test = st.session_state['test_data']
        
        # Model comparison
        st.subheader("🏆 Model Performance")
        
        perf_data = []
        for name, result in results.items():
            perf_data.append({
                'Model': name,
                'AUC Score': result['auc'],
                'Accuracy': np.mean(result['predictions'] == y_test)
            })
        
        perf_df = pd.DataFrame(perf_data).sort_values('AUC Score', ascending=False)
        st.dataframe(perf_df)
        
        # Best model details
        best_model_name = perf_df.iloc[0]['Model']
        best_result = results[best_model_name]
        
        st.subheader(f"📈 {best_model_name} - Detailed Results")
        
        # Classification report
        report = classification_report(y_test, best_result['predictions'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Feature importance (if available)
        if hasattr(best_result['model'], 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            feature_imp = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': best_result['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_imp.head(10), x='Importance', y='Feature', 
                        title="Top 10 Most Important Features",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)

def show_admin_panel(data_manager):
    """Show admin panel (admin only)"""
    st.header("⚙️ Admin Panel")
    
    if st.session_state['user_role'] != 'admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    # System statistics
    st.subheader("📊 System Statistics")
    
    shared_datasets = data_manager.get_shared_datasets()
    all_user_datasets = data_manager.metadata.get("user_datasets", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", len(all_user_datasets))
    with col2:
        st.metric("Shared Datasets", len(shared_datasets))
    with col3:
        total_user_datasets = sum(len(datasets) for datasets in all_user_datasets.values())
        st.metric("User Datasets", total_user_datasets)
    
    # Manage shared datasets
    st.subheader("🗂️ Manage Shared Datasets")
    
    if shared_datasets:
        for filename, metadata in shared_datasets.items():
            with st.expander(f"📁 {metadata['name']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Uploader:** {metadata['uploader']}")
                    st.write(f"**Description:** {metadata['description']}")
                    st.write(f"**Upload Date:** {metadata['upload_date']}")
                    st.write(f"**Size:** {metadata['file_size_mb']:.1f} MB")
                
                with col2:
                    if st.button("🗑️ Delete", key=f"admin_delete_{filename}"):
                        success, message = data_manager.delete_dataset('shared', None, filename)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
    else:
        st.info("No shared datasets to manage.")

if __name__ == "__main__":
    main()