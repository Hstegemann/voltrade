import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

st.set_page_config(page_title="Volatility Trading ML", layout="wide")

st.title("🚀 Volatility Trading ML Dashboard")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Data Processing", "Analysis", "Model Training"])

if page == "Data Processing":
    st.header("📊 Data Processing")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your merged dataset CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Read the data
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows of data")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Process button
        if st.button("🔧 Process VIX Futures Data"):
            with st.spinner("Processing data..."):
                try:
                    # Parse dates
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date')
                    
                    # Find VIX futures columns
                    vix_futures = {}
                    for col in df.columns:
                        if 'VIX_Close' in col and '-' in col:
                            contract = col.split('-')[1]
                            vix_futures[contract] = col
                    
                    st.write(f"Found {len(vix_futures)} VIX futures contracts")
                    
                    # Create continuous contract
                    continuous_prices = []
                    
                    # Simple approach - use the first non-null value across contracts
                    for idx in df.index:
                        price = None
                        for contract, col in vix_futures.items():
                            val = df.loc[idx, col]
                            if pd.notna(val) and val > 0:
                                price = val
                                break
                        continuous_prices.append(price)
                    
                    # Create synthetic VIXY
                    continuous = pd.Series(continuous_prices, index=df.index).dropna()
                    
                    # Calculate returns
                    returns = continuous.pct_change()
                    
                    # Create synthetic VIXY with decay
                    vixy_prices = [20.0]  # Starting price
                    daily_decay = -0.0025
                    
                    for i in range(1, len(returns)):
                        if pd.notna(returns.iloc[i]):
                            vixy_return = returns.iloc[i] * 0.5 + daily_decay
                            new_price = vixy_prices[-1] * (1 + vixy_return)
                            vixy_prices.append(new_price)
                    
                    # Create simplified dataset
                    simple_df = pd.DataFrame(index=continuous.index[:len(vixy_prices)])
                    simple_df['VIX_Close'] = continuous.iloc[:len(vixy_prices)]
                    simple_df['VIXY_Close'] = vixy_prices
                    
                    # Add other assets if available
                    for asset in ['SPX', 'HYG', 'TLT', 'QQQ']:
                        asset_col = next((col for col in df.columns if f'{asset}_Close' in col), None)
                        if asset_col:
                            simple_df[f'{asset}_Close'] = df.loc[simple_df.index, asset_col]
                    
                    # Show results
                    st.success("✅ Data processing complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("VIX Continuous Contract")
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        simple_df['VIX_Close'].plot(ax=ax1)
                        ax1.set_title("VIX Continuous")
                        ax1.set_ylabel("Level")
                        st.pyplot(fig1)
                    
                    with col2:
                        st.subheader("Synthetic VIXY")
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        simple_df['VIXY_Close'].plot(ax=ax2, color='green')
                        ax2.set_title("VIXY (Synthetic)")
                        ax2.set_ylabel("Price ($)")
                        st.pyplot(fig2)
                    
                    # Download button
                    csv = simple_df.to_csv()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="📥 Download Processed Data",
                        data=csv,
                        file_name=f'simplified_volatility_data_{timestamp}.csv',
                        mime='text/csv'
                    )
                    
                    # Store in session state
                    st.session_state['processed_data'] = simple_df
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    else:
        st.info("👆 Upload your merged dataset to begin")
        
        # Option to load the file directly if it exists
        if st.button("Load merged_multi_asset_dataset_20250714_112740.csv"):
            try:
                df = pd.read_csv('merged_multi_asset_dataset_20250714_112740.csv')
                st.success("File loaded from disk!")
                st.experimental_rerun()
            except:
                st.error("File not found in current directory")

elif page == "Analysis":
    st.header("📈 Data Analysis")
    
    if 'processed_data' in st.session_state:
        df = st.session_state['processed_data']
        
        # Show statistics
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ Please process data first")

elif page == "Model Training":
    st.header("🤖 Model Training")
    st.info("Model training functionality coming soon...")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Volatility Trading ML Project")
st.sidebar.markdown("[GitHub Repository](https://github.com/hstegemann/voltrade)")
