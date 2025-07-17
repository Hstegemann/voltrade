# auth_config.py
"""
Authentication Configuration for ML Volatility Trading Model
⚠️ IMPORTANT: Change these credentials before deploying!
"""
import hashlib
import os

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(str.encode(password)).hexdigest()

# =====================================================================
# USER CREDENTIALS - CHANGE THESE BEFORE DEPLOYMENT!
# =====================================================================

# Option 1: Define users directly (for development)
USERS = {
    "admin": hash_password("VixTrading2025!"),       # Full access admin
    "trader": hash_password("TradingSecure123"),     # Trading access
    "analyst": hash_password("AnalysisView456"),     # View-only access
    "james": hash_password("YourPersonalPass"),      # Your personal account
}

# Option 2: Use environment variables (for production)
USERS_FROM_ENV = {
    "admin": hash_password(os.getenv("ADMIN_PASSWORD", "VixTrading2025!")),
    "trader": hash_password(os.getenv("TRADER_PASSWORD", "TradingSecure123")),
    "analyst": hash_password(os.getenv("ANALYST_PASSWORD", "AnalysisView456")),
    "james": hash_password(os.getenv("JAMES_PASSWORD", "YourPersonalPass")),
}

# Choose which method to use
USE_ENVIRONMENT_VARIABLES = False  # Set to True for production

# Final user configuration
FINAL_USERS = USERS_FROM_ENV if USE_ENVIRONMENT_VARIABLES else USERS

# =====================================================================
# USER PERMISSIONS
# =====================================================================

USER_PERMISSIONS = {
    "admin": ["upload", "train", "download", "view", "manage"],
    "trader": ["upload", "train", "view", "download"],
    "analyst": ["view", "download"],
    "james": ["upload", "train", "download", "view", "manage"],  # Your full access
}

# =====================================================================
# SECURITY SETTINGS
# =====================================================================

# Session timeout (in seconds) - set to None for no timeout
SESSION_TIMEOUT = 3600  # 1 hour

# Maximum failed login attempts
MAX_LOGIN_ATTEMPTS = 5

# Lock account duration after max attempts (in seconds)
LOCKOUT_DURATION = 300  # 5 minutes

# =====================================================================
# DEPLOYMENT CONFIGURATION
# =====================================================================

# Set this to True when deploying to production
PRODUCTION_MODE = False

# In production, disable demo credentials display
SHOW_DEMO_CREDENTIALS = not PRODUCTION_MODE

# Enable logging of authentication attempts
LOG_AUTH_ATTEMPTS = True