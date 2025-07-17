#!/usr/bin/env python3
"""
ML Volatility Trading Model - Launch Script
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n💡 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launch function"""
    print("🚀 ML Volatility Trading Model")
    print("=" * 50)
    
    # Check if streamlit_app.py exists
    if not os.path.exists('streamlit_app.py'):
        print("❌ streamlit_app.py not found!")
        print("Please make sure the main app file is in the current directory.")
        return
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    print("✅ All requirements satisfied!")
    print("\n🌐 Starting Streamlit app...")
    print("📱 App will open in your browser automatically")
    print("🔗 Manual URL: http://localhost:8501")
    print("\n⚠️  To stop the app, press Ctrl+C in this terminal")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")

if __name__ == "__main__":
    main()