import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import os
import glob
import warnings
import pytz
import re
warnings.filterwarnings('ignore')

class FuturesExpirationCalculator:
    """Enhanced calculator for futures expiration dates - ALL TIMES IN EASTERN"""
    
    def __init__(self):
        # Set up timezone handling - all times will be converted to Eastern
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.central_tz = pytz.timezone('US/Central')
        
        self.month_codes = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
        
        # Futures type specifications - ALL TIMES CONVERTED TO EASTERN WITH PROPER DST HANDLING
        self.futures_specs = {
            'VI': {  # VIX (CBOE - HARDCODED to 9:00 AM ET for consistency)
                'day_rule': 'third_wednesday_minus_30',
                'native_time': (9, 0),  # HARDCODED: 9:00 AM ET (no conversion)
                'native_timezone': 'US/Eastern',  # Treat as Eastern directly
                'description': 'VIX futures expire 30 days before 3rd Friday of following month at 9:00 AM ET (hardcoded)'
            },
            'ES': {  # E-mini S&P 500 (CME - Central time, converted to Eastern)
                'day_rule': 'third_friday',
                'native_time': (9, 30),  # 9:30 AM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': 'E-mini S&P 500 expires 3rd Friday at 9:30 AM Central Time'
            },
            'NQ': {  # NASDAQ (CME - Central time, converted to Eastern)
                'day_rule': 'third_friday',
                'native_time': (9, 30),  # 9:30 AM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': 'NASDAQ expires 3rd Friday at 9:30 AM Central Time'
            },
            'QR': {  # Russell 2000 (CME - Central time, converted to Eastern)
                'day_rule': 'third_friday',
                'native_time': (9, 30),  # 9:30 AM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': 'Russell 2000 expires 3rd Friday at 9:30 AM Central Time'
            },
            'YM': {  # Dow Jones (CME - Central time, converted to Eastern)
                'day_rule': 'third_friday',
                'native_time': (9, 30),  # 9:30 AM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': 'Dow Jones expires 3rd Friday at 9:30 AM Central Time'
            },
            'CL': {  # Crude Oil (NYMEX - Eastern time natively)
                'day_rule': 'third_business_day_before_25th',
                'native_time': (14, 30),  # 2:30 PM ET (native)
                'native_timezone': 'US/Eastern',
                'description': 'Crude Oil expires 3rd business day before 25th at 2:30 PM ET'
            },
            'GC': {  # Gold (COMEX - Eastern time natively)
                'day_rule': 'third_last_business_day',
                'native_time': (13, 30),  # 1:30 PM ET (native)
                'native_timezone': 'US/Eastern',
                'description': 'Gold expires 3rd last business day of month at 1:30 PM ET'
            },
            'ZB': {  # 30-Year Treasury Bond (CBOT - Central time, converted to Eastern)
                'day_rule': 'third_friday',
                'native_time': (12, 0),   # 12:00 PM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': 'Treasury Bond expires 3rd Friday at 12:00 PM Central Time'
            },
            'ZN': {  # 10-Year Treasury Note (CBOT - Central time, converted to Eastern)
                'day_rule': 'third_friday', 
                'native_time': (12, 0),   # 12:00 PM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': 'Treasury Note expires 3rd Friday at 12:00 PM Central Time'
            },
            'ZF': {  # 5-Year Treasury Note (CBOT - Central time, converted to Eastern)
                'day_rule': 'third_friday',
                'native_time': (12, 0),   # 12:00 PM Central (CST/CDT depending on date)
                'native_timezone': 'US/Central',
                'description': '5-Year Treasury Note expires 3rd Friday at 12:00 PM Central Time'
            }
        }
        
        # Exchange timezone mappings for reference
        self.exchange_timezones = {
            'CBOE': 'US/Eastern',      # VIX futures - HARDCODED to 9:00 AM ET
            'CME': 'US/Central',       # ES, NQ, YM, QR
            'CBOT': 'US/Central',      # ZB, ZN, ZF, ZT (Treasury futures)
            'NYMEX': 'US/Eastern',     # CL, NG, HO, RB
            'COMEX': 'US/Eastern',     # GC, SI, HG, PL
        }
    
    def _create_eastern_datetime(self, year: int, month: int, day: int, hour: int, minute: int, native_timezone: str = 'US/Eastern') -> datetime:
        """Create a datetime object in specified timezone, then convert to Eastern"""
        if native_timezone == 'US/Central':
            # Create in Central time first (this handles DST automatically)
            central_tz = pytz.timezone('US/Central')
            naive_dt = datetime(year, month, day, hour, minute)
            central_dt = central_tz.localize(naive_dt)
            # Convert to Eastern
            return central_dt.astimezone(self.eastern_tz)
        else:
            # Create directly in Eastern
            naive_dt = datetime(year, month, day, hour, minute)
            return self.eastern_tz.localize(naive_dt)
    
    def _create_native_datetime_then_convert(self, year: int, month: int, day: int, hour: int, minute: int, native_timezone: str) -> datetime:
        """Create datetime in native timezone first, then convert to Eastern - handles DST properly"""
        naive_dt = datetime(year, month, day, hour, minute)
        
        if native_timezone == 'US/Central':
            # Use Central timezone (automatically handles CST vs CDT)
            native_tz = pytz.timezone('US/Central')
            native_dt = native_tz.localize(naive_dt)
            return native_dt.astimezone(self.eastern_tz)
        elif native_timezone == 'US/Eastern':
            # Already Eastern
            return self.eastern_tz.localize(naive_dt)
        else:
            # Other timezone
            native_tz = pytz.timezone(native_timezone)
            native_dt = native_tz.localize(naive_dt)
            return native_dt.astimezone(self.eastern_tz)
    
    def get_third_friday(self, year, month):
        """Get the third Friday of a month"""
        c = calendar.monthcalendar(year, month)
        fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
        return fridays[2] if len(fridays) >= 3 else None
    
    def get_third_wednesday(self, year, month):
        """Get the third Wednesday of a month"""
        c = calendar.monthcalendar(year, month)
        wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
        return wednesdays[2] if len(wednesdays) >= 3 else None
    
    def get_business_days_before_date(self, year, month, target_day, days_before):
        """Get business day N days before target date"""
        target_date = datetime(year, month, target_day)
        current_date = target_date
        business_days_count = 0
        
        while business_days_count < days_before:
            current_date -= timedelta(days=1)
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                business_days_count += 1
        
        return current_date.day
    
    def get_last_business_days(self, year, month, n_days=3):
        """Get the nth last business day of month"""
        # Start from last day of month
        last_day = calendar.monthrange(year, month)[1]
        current_date = datetime(year, month, last_day)
        business_days_count = 0
        
        while business_days_count < n_days:
            if current_date.weekday() < 5:  # Is business day
                business_days_count += 1
                if business_days_count == n_days:
                    return current_date.day
            current_date -= timedelta(days=1)
        
        return None
    
    def calculate_expiration(self, contract_id, futures_type=None):
        """Calculate expiration date for any futures contract - returns Eastern Time"""
        if not contract_id or len(contract_id) < 3:
            return None
        
        try:
            # Extract month and year
            month_char = contract_id[-3]
            year_str = contract_id[-2:]
            
            if month_char not in self.month_codes:
                return None
            
            month = self.month_codes[month_char]
            year = 2000 + int(year_str)
            
            # Auto-detect futures type if not provided
            if not futures_type:
                if contract_id.startswith('VI'):
                    futures_type = 'VI'
                elif contract_id.startswith('ES'):
                    futures_type = 'ES'
                elif contract_id.startswith('QR'):
                    futures_type = 'QR'
                elif contract_id.startswith('NQ'):
                    futures_type = 'NQ'
                elif contract_id.startswith('YM'):
                    futures_type = 'YM'
                elif contract_id.startswith('CL'):
                    futures_type = 'CL'
                elif contract_id.startswith('GC'):
                    futures_type = 'GC'
                elif contract_id.startswith('ZB'):
                    futures_type = 'ZB'
                elif contract_id.startswith('ZN'):
                    futures_type = 'ZN'
                elif contract_id.startswith('ZF'):
                    futures_type = 'ZF'
                else:
                    # Default to equity index rules
                    futures_type = 'ES'
            
            if futures_type not in self.futures_specs:
                futures_type = 'ES'  # Default fallback
            
            spec = self.futures_specs[futures_type]
            hour, minute = spec['native_time']  # Native exchange time
            native_timezone = spec['native_timezone']
            
            # Calculate expiration day based on rule
            if spec['day_rule'] == 'third_friday':
                day = self.get_third_friday(year, month)
            elif spec['day_rule'] == 'third_wednesday':
                day = self.get_third_wednesday(year, month)
            elif spec['day_rule'] == 'third_wednesday_minus_30':
                # VIX special rule: 30 days before 3rd Friday of FOLLOWING month
                next_month = month + 1 if month < 12 else 1
                next_year = year if month < 12 else year + 1
                third_friday = self.get_third_friday(next_year, next_month)
                if third_friday:
                    target_date = self._create_native_datetime_then_convert(
                        next_year, next_month, third_friday, hour, minute, native_timezone
                    )
                    # Calculate the target expiry date (30 days before)
                    target_expiry_date = target_date - timedelta(days=30)
                    
                    # FIXED: Create the expiry with proper timezone for the actual expiry date
                    # Don't inherit timezone from target_date, create fresh for the expiry date
                    expiry_date = self._create_native_datetime_then_convert(
                        target_expiry_date.year, target_expiry_date.month, target_expiry_date.day,
                        hour, minute, native_timezone
                    )
                    return expiry_date
            elif spec['day_rule'] == 'third_business_day_before_25th':
                day = self.get_business_days_before_date(year, month, 25, 3)
            elif spec['day_rule'] == 'third_last_business_day':
                day = self.get_last_business_days(year, month, 3)
            else:
                day = self.get_third_friday(year, month)  # Default
            
            if day:
                return self._create_native_datetime_then_convert(
                    year, month, day, hour, minute, native_timezone
                )
                
        except Exception as e:
            print(f"Error calculating expiration for {contract_id}: {e}")
        
        return None

class ContinuousFuturesCalculator:
    """Calculator for continuous futures contracts (synthetic constant maturity)"""
    
    def __init__(self, target_days_to_expiry=30):
        self.target_days = target_days_to_expiry
        
    def calculate_continuous_price(self, front_price, next_price, front_days, next_days):
        """
        Calculate continuous futures price using linear interpolation
        Based on your handwritten notes:
        - Target: 30-day synthetic contract
        - Linear interpolation between front and next month
        """
        if front_price is None or next_price is None:
            return None
            
        if front_days is None or next_days is None:
            return None
            
        # Ensure we have valid data
        if front_days <= 0 or next_days <= front_days:
            return front_price  # Use front month if expiry logic is off
            
        # Calculate weights for interpolation
        # Weight = (target_days - front_days) / (next_days - front_days)
        if next_days == front_days:
            return front_price  # Avoid division by zero
            
        # Linear interpolation weight
        weight = (self.target_days - front_days) / (next_days - front_days)
        
        # Clamp weight between 0 and 1
        weight = max(0, min(1, weight))
        
        # Calculate continuous price
        continuous_price = front_price * (1 - weight) + next_price * weight
        
        return continuous_price
    
    def calculate_weights(self, front_days, next_days):
        """Calculate the weights for front and next month contracts"""
        if front_days is None or next_days is None:
            return None, None
            
        if next_days == front_days:
            return 1.0, 0.0  # All front month
            
        # Calculate interpolation weight for next month
        next_weight = (self.target_days - front_days) / (next_days - front_days)
        next_weight = max(0, min(1, next_weight))
        
        # Front month weight is complement
        front_weight = 1 - next_weight
        
        return front_weight, next_weight

# Enhanced helper functions for missing data recovery

def find_contract_price_enhanced(contract_id, row, contract_cols, contract_info, futures_df=None, current_date=None):
    """
    Enhanced price finder with multiple strategies - FIXES MISSING PRICE DATA
    """
    if not contract_id:
        return None
    
    # STRATEGY 1: Direct column match in current row
    for col in contract_cols:
        if col in contract_info:
            futures_type, col_contract_id = contract_info[col]
            if col_contract_id == contract_id:
                try:
                    price_val = row[col]
                    if isinstance(price_val, str):
                        price_val = str(price_val).replace(',', '')
                    if not pd.isna(price_val) and price_val != '' and str(price_val) != '0':
                        return float(price_val)
                except:
                    continue
    
    # STRATEGY 2: Look for partial column name matches
    contract_suffix = contract_id[-3:]  # Get month+year (e.g., "G25")
    for col in contract_cols:
        if contract_suffix in col.upper():
            try:
                price_val = row[col]
                if isinstance(price_val, str):
                    price_val = str(price_val).replace(',', '')
                if not pd.isna(price_val) and price_val != '' and str(price_val) != '0':
                    return float(price_val)
            except:
                continue
    
    # STRATEGY 3: Look in nearby time periods if futures_df provided
    if futures_df is not None and current_date is not None:
        try:
            # Look ±5 minutes around current time
            time_window = pd.Timedelta(minutes=5)
            start_time = current_date - time_window
            end_time = current_date + time_window
            
            # Filter data in time window
            unified_date_col = 'unified_date' if 'unified_date' in futures_df.columns else futures_df.columns[0]
            time_mask = (futures_df[unified_date_col] >= start_time) & (futures_df[unified_date_col] <= end_time)
            nearby_data = futures_df[time_mask]
            
            for col in contract_cols:
                if col in contract_info:
                    futures_type, col_contract_id = contract_info[col]
                    if col_contract_id == contract_id and col in nearby_data.columns:
                        # Find last valid price in window
                        valid_prices = nearby_data[col].dropna()
                        valid_prices = valid_prices[valid_prices != 0]
                        if len(valid_prices) > 0:
                            return float(valid_prices.iloc[-1])
        except:
            pass
    
    return None

def find_next_month_contract_enhanced(current_front, active_contracts, current_date, calc, main_futures_type):
    """
    Enhanced next month contract finder with multiple fallback strategies - FIXES MISSING NEXT MONTH
    """
    if not current_front or not active_contracts:
        return None
    
    # METHOD 1: Month sequence logic (existing)
    month_sequence = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    
    if len(current_front) >= 3:
        front_month_char = current_front[-3]
        front_year = current_front[-2:]
        
        if front_month_char in month_sequence:
            current_pos = month_sequence.index(front_month_char)
            
            # Try next months in sequence (same year)
            for next_pos in range(current_pos + 1, len(month_sequence)):
                next_month_char = month_sequence[next_pos]
                next_contract_id = current_front[:-3] + next_month_char + front_year
                
                # Check if this contract exists in active contracts
                for contract in active_contracts:
                    if contract['contract_id'] == next_contract_id:
                        return next_contract_id
            
            # Try next year contracts
            try:
                next_year = str(int(front_year) + 1)[-2:]
                for next_month_char in month_sequence:
                    next_contract_id = current_front[:-3] + next_month_char + next_year
                    
                    for contract in active_contracts:
                        if contract['contract_id'] == next_contract_id:
                            return next_contract_id
            except:
                pass
    
    # METHOD 2: Find second earliest expiring contract
    if len(active_contracts) >= 2:
        # Sort by expiry and take second
        try:
            sorted_contracts = sorted(active_contracts, key=lambda x: x['expiry'])
            if len(sorted_contracts) >= 2:
                return sorted_contracts[1]['contract_id']
        except:
            pass
    
    # METHOD 3: Find any contract that's not the front month
    for contract in active_contracts:
        if contract['contract_id'] != current_front:
            return contract['contract_id']
    
    return None

def get_contract_data_robust(contract_id, row, contract_cols, contract_info, calc, 
                           main_futures_type, current_date, futures_df=None):
    """
    Get comprehensive contract data with enhanced price finding
    """
    if not contract_id:
        return None
    
    # Get expiry
    expiry = calc.calculate_expiration(contract_id, main_futures_type)
    if not expiry:
        return None
    
    # Check if expired
    has_expired = has_contract_expired_strict(expiry, current_date)
    if has_expired:
        return None
    
    # Get price using enhanced finder
    price = find_contract_price_enhanced(
        contract_id, row, contract_cols, contract_info, futures_df, current_date
    )
    
    # Calculate days to expiry
    try:
        is_active, time_diff = safe_datetime_compare(expiry, current_date)
        days_to_exp = time_diff.total_seconds() / 86400
    except:
        days_to_exp = None
    
    return {
        'contract_id': contract_id,
        'expiry': expiry,
        'price': price,
        'days': days_to_exp
    }

# Utility functions
def extract_contract_info(contract_name):
    """Extract contract type and month-year from column name"""
    contract_name = str(contract_name).upper()
    
    # Try different patterns
    patterns = [
        r'(VI|ES|QR|NQ|YM|RTY|CL|GC|SI|ZB|ZN|ZF|ZT)([FGHJKMNQUVXZ]\d{2})',  # Standard pattern
        r'([A-Z]{2,3})([FGHJKMNQUVXZ]\d{2})',  # Any 2-3 letter prefix
    ]
    
    for pattern in patterns:
        match = re.search(pattern, contract_name)
        if match:
            return match.group(1), match.group(1) + match.group(2)
    
    return None, None

def detect_futures_type(contract_cols):
    """Detect what type of futures we're dealing with"""
    futures_types = set()
    for col in contract_cols[:5]:  # Check first 5 contracts
        futures_type, _ = extract_contract_info(col)
        if futures_type:
            futures_types.add(futures_type)
    
    if len(futures_types) == 1:
        return list(futures_types)[0]
    elif 'VI' in futures_types:
        return 'VI'  # Prefer VIX if mixed
    elif futures_types:
        return list(futures_types)[0]
    return 'UNKNOWN'

def detect_tickers_from_data(spot_col, main_futures_type):
    """Extract actual ticker symbols from column names"""
    
    # Extract spot ticker from spot column name
    spot_ticker = "SPX"  # Default
    if 'SPX' in spot_col.upper():
        spot_ticker = "SPX"
    elif 'VIX' in spot_col.upper():
        spot_ticker = "VIX"
    elif 'IWM' in spot_col.upper():
        spot_ticker = "IWM"
    elif 'QQQ' in spot_col.upper():
        spot_ticker = "QQQ"
    else:
        # Try to extract from column name
        for part in spot_col.split('_'):
            if len(part) >= 2 and part.isalpha():
                spot_ticker = part.upper()
                break
    
    # Extract futures ticker from futures type
    futures_ticker = main_futures_type
    if main_futures_type == 'VI':
        futures_ticker = "VIX"  # VIX futures
    elif main_futures_type == 'ES':
        futures_ticker = "ES"   # E-mini S&P 500
    elif main_futures_type == 'QR':
        futures_ticker = "RTY"  # Russell 2000 (sometimes called RTY)
    elif main_futures_type == 'NQ':
        futures_ticker = "NQ"   # NASDAQ
    elif main_futures_type == 'YM':
        futures_ticker = "YM"   # Dow Jones
    elif main_futures_type == 'CL':
        futures_ticker = "CL"   # Crude Oil
    elif main_futures_type == 'GC':
        futures_ticker = "GC"   # Gold
    else:
        futures_ticker = main_futures_type
    
    return spot_ticker, futures_ticker

def format_datetime_consistent(dt):
    """Format datetime consistently: M/D/YYYY H:MM (no timezone suffix)"""
    if pd.isna(dt):
        return ''
    try:
        # Handle timezone-aware datetimes
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            # Convert to Eastern if not already
            eastern_tz = pytz.timezone('US/Eastern')
            if dt.tzinfo != eastern_tz:
                dt = dt.astimezone(eastern_tz)
        
        month = dt.month
        day = dt.day
        year = dt.year
        hour = dt.hour
        minute = dt.minute
        return f"{month}/{day}/{year} {hour}:{minute:02d}"
    except:
        return str(dt)

def standardize_datetime_to_eastern(dt_series, futures_type=None):
    """Standardize datetime series to Eastern Time"""
    eastern_tz = pytz.timezone('US/Eastern')
    
    def convert_single_datetime(dt):
        if pd.isna(dt):
            return dt
        
        # Convert to pandas datetime first if it's a string
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        # Ensure it's a datetime object
        if not isinstance(dt, (datetime, pd.Timestamp)):
            dt = pd.to_datetime(dt)
        
        # If already timezone-aware, convert to Eastern
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.astimezone(eastern_tz)
        
        # If naive datetime, assume it's already in Eastern (since user said files are adjusted)
        # But apply timezone awareness
        return eastern_tz.localize(dt)
    
    return dt_series.apply(convert_single_datetime)

def safe_datetime_compare(expiry_dt, current_dt):
    """Safely compare timezone-aware and potentially naive datetimes"""
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Convert current_dt to datetime if it's a string
    if isinstance(current_dt, str):
        current_dt = pd.to_datetime(current_dt)
    elif not isinstance(current_dt, (datetime, pd.Timestamp)):
        current_dt = pd.to_datetime(current_dt)
    
    # Ensure both datetimes are timezone-aware
    if hasattr(current_dt, 'tzinfo') and current_dt.tzinfo is not None:
        current_aware = current_dt
    else:
        # Convert naive datetime to Eastern
        current_aware = eastern_tz.localize(current_dt)
    
    if hasattr(expiry_dt, 'tzinfo') and expiry_dt.tzinfo is not None:
        expiry_aware = expiry_dt
    else:
        # This shouldn't happen with our expiry dates, but just in case
        expiry_aware = eastern_tz.localize(expiry_dt)
    
    return expiry_aware > current_aware, expiry_aware - current_aware

def has_contract_expired_strict(expiry_dt, current_dt):
    """
    STRICT expiration check - returns True AT OR AFTER the exact expiration time
    Special handling: If expiry is 9:00 AM, trigger on any timestamp in the 9:00 AM hour
    This ensures rollovers happen at 9:00 sharp, not 9:01
    """
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Convert current_dt to datetime if it's a string
    if isinstance(current_dt, str):
        current_dt = pd.to_datetime(current_dt)
    elif not isinstance(current_dt, (datetime, pd.Timestamp)):
        current_dt = pd.to_datetime(current_dt)
    
    # Ensure both datetimes are timezone-aware
    if hasattr(current_dt, 'tzinfo') and current_dt.tzinfo is not None:
        current_aware = current_dt
    else:
        # Convert naive datetime to Eastern
        current_aware = eastern_tz.localize(current_dt)
    
    if hasattr(expiry_dt, 'tzinfo') and expiry_dt.tzinfo is not None:
        expiry_aware = expiry_dt
    else:
        # This shouldn't happen with our expiry dates, but just in case
        expiry_aware = eastern_tz.localize(expiry_dt)
    
    # SPECIAL VIX HANDLING: If expiry is at 9:00 AM, trigger on any timestamp in the 9:00 AM hour
    if (expiry_aware.hour == 9 and expiry_aware.minute == 0 and 
        current_aware.date() == expiry_aware.date() and current_aware.hour == 9):
        return True  # Any 9:xx AM timestamp triggers expiry for 9:00 AM expiry
    
    # Standard check: Contract has expired AT OR AFTER the exact expiration time
    return current_aware >= expiry_aware

def display_calendar_with_highlight(year, month, highlight_day, contract_id, expiry_time):
    """Display a calendar with the expiration day highlighted"""
    import calendar as cal
    
    # Month names
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    print(f"\n📅 EXPIRATION CALENDAR FOR {contract_id}")
    print("=" * 50)
    print(f"{month_names[month-1]} {year}")
    print("=" * 50)
    
    # Get calendar for the month
    cal_matrix = cal.monthcalendar(year, month)
    
    # Header
    print("Mo Tu We Th Fr Sa Su")
    
    # Display each week
    for week in cal_matrix:
        week_str = ""
        for day in week:
            if day == 0:
                week_str += "   "  # Empty day
            elif day == highlight_day:
                week_str += f"[{day:2d}]"  # Highlighted day with brackets
            else:
                week_str += f" {day:2d} "  # Regular day
        print(week_str)
    
    print("\n" + "=" * 50)
    print(f"📍 Highlighted: {month_names[month-1]} {highlight_day}, {year}")
    print(f"⏰ Expiration: {expiry_time.strftime('%I:%M %p %Z')}")
    print(f"📊 Contract: {contract_id}")
    print("=" * 50)

def verify_expiration_dates(calc, contract_cols, contract_info):
    """Interactive verification of expiration dates"""
    print("\n" + "="*60)
    print("EXPIRATION DATE VERIFICATION")
    print("="*60)
    print("Please verify that the calculated expiration dates are correct.")
    print("This helps ensure accurate rollover timing.")
    print("")
    
    # Get unique contracts and sort by expiration
    contracts_to_verify = []
    shown_contracts = set()
    
    for col in contract_cols[:10]:  # Check first 10
        if col in contract_info:
            futures_type, contract_id = contract_info[col]
            if contract_id not in shown_contracts:
                expiry = calc.calculate_expiration(contract_id, futures_type)
                if expiry:
                    contracts_to_verify.append({
                        'contract_id': contract_id,
                        'futures_type': futures_type,
                        'expiry': expiry
                    })
                    shown_contracts.add(contract_id)
    
    # Sort by expiration date
    contracts_to_verify.sort(key=lambda x: x['expiry'])
    
    verified_contracts = []
    
    for i, contract in enumerate(contracts_to_verify):
        contract_id = contract['contract_id']
        futures_type = contract['futures_type']
        expiry = contract['expiry']
        
        print(f"\nContract {i+1} of {len(contracts_to_verify)}: {contract_id}")
        print("-" * 40)
        
        # Display the calendar
        display_calendar_with_highlight(
            expiry.year, 
            expiry.month, 
            expiry.day, 
            contract_id, 
            expiry
        )
        
        # Get user confirmation
        while True:
            response = input(f"\nIs this expiration correct for {contract_id}? (y/n/s=skip): ").lower().strip()
            
            if response in ['y', 'yes']:
                verified_contracts.append(contract)
                print(f"✅ {contract_id} expiration confirmed")
                break
            elif response in ['n', 'no']:
                print(f"\n🔧 Manual correction for {contract_id}")
                corrected_expiry = get_manual_expiration_input(contract_id, expiry)
                if corrected_expiry:
                    contract['expiry'] = corrected_expiry
                    verified_contracts.append(contract)
                    print(f"✅ {contract_id} expiration manually set")
                break
            elif response in ['s', 'skip']:
                print(f"⚠️ Skipping {contract_id} verification")
                verified_contracts.append(contract)  # Keep original
                break
            else:
                print("Please enter 'y' for yes, 'n' for no, or 's' to skip")
    
    print(f"\n✅ Verification complete! {len(verified_contracts)} contracts verified.")
    return verified_contracts

def get_manual_expiration_input(contract_id, current_expiry):
    """Get manual expiration date/time input from user"""
    print(f"\nCurrent expiration for {contract_id}:")
    print(f"Date: {current_expiry.strftime('%m/%d/%Y')}")
    print(f"Time: {current_expiry.strftime('%I:%M %p %Z')}")
    
    try:
        # Get new date
        while True:
            date_input = input("Enter new date (MM/DD/YYYY) or press Enter to keep current: ").strip()
            if not date_input:
                new_date = current_expiry.date()
                break
            try:
                from datetime import datetime
                new_date = datetime.strptime(date_input, '%m/%d/%Y').date()
                break
            except ValueError:
                print("Invalid date format. Please use MM/DD/YYYY")
        
        # Get new time
        while True:
            time_input = input("Enter new time (HH:MM AM/PM) or press Enter to keep current: ").strip()
            if not time_input:
                new_time = current_expiry.time()
                break
            try:
                # Try different time formats
                for fmt in ['%I:%M %p', '%H:%M', '%I:%M%p']:
                    try:
                        new_time = datetime.strptime(time_input, fmt).time()
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError("Invalid time format")
                break
            except ValueError:
                print("Invalid time format. Please use HH:MM AM/PM (e.g., 10:30 AM)")
        
        # Combine date and time
        eastern_tz = pytz.timezone('US/Eastern')
        new_datetime = datetime.combine(new_date, new_time)
        new_expiry = eastern_tz.localize(new_datetime)
        
        print(f"New expiration: {new_expiry.strftime('%m/%d/%Y %I:%M %p %Z')}")
        
        confirm = input("Confirm this new expiration? (y/n): ").lower().strip()
        if confirm in ['y', 'yes']:
            return new_expiry
        else:
            print("Keeping original expiration")
            return None
            
    except Exception as e:
        print(f"Error setting manual expiration: {e}")
        return None

def ask_verification_preference():
    """Ask user if they want to verify expiration dates"""
    print("\n" + "="*60)
    print("EXPIRATION DATE VERIFICATION OPTION")
    print("="*60)
    print("Would you like to verify the calculated expiration dates?")
    print("This will show a calendar for each contract's expiration date.")
    print("Recommended for first-time use or when using new contracts.")
    print("")
    
    while True:
        response = input("Verify expiration dates? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

def update_calculator_with_verified_dates(calc, verified_contracts):
    """Update the calculator with verified expiration dates"""
    # Create a dictionary of verified expirations
    verified_expirations = {}
    for contract in verified_contracts:
        verified_expirations[contract['contract_id']] = contract['expiry']
    
    # Store in calculator for later use
    calc.verified_expirations = verified_expirations
    
    # Update the calculate_expiration method to use verified dates
    original_calculate = calc.calculate_expiration
    
    def calculate_with_verification(contract_id, futures_type=None):
        if hasattr(calc, 'verified_expirations') and contract_id in calc.verified_expirations:
            return calc.verified_expirations[contract_id]
        else:
            return original_calculate(contract_id, futures_type)
    
    calc.calculate_expiration = calculate_with_verification

def browse_directories(start_path=None):
    """Browse directories interactively"""
    if start_path is None:
        # Start from user's home directory
        start_path = os.path.expanduser("~")
    
    current_path = os.path.abspath(start_path)
    
    while True:
        print(f"\n📁 Current location: {current_path}")
        print("=" * 60)
        
        try:
            # Get all directories in current path
            items = []
            
            # Add parent directory option (except for root)
            parent_dir = os.path.dirname(current_path)
            if parent_dir != current_path:  # Not at root
                items.append(("📁 ..", parent_dir, "directory"))
            
            # Get subdirectories
            try:
                for item in sorted(os.listdir(current_path)):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        # Check if directory contains CSV or Excel files
                        try:
                            csv_files = len([f for f in os.listdir(item_path) 
                                           if f.lower().endswith(('.csv', '.xlsx', '.xls'))])
                            file_indicator = f" ({csv_files} data files)" if csv_files > 0 else ""
                            items.append((f"📁 {item}{file_indicator}", item_path, "directory"))
                        except PermissionError:
                            items.append((f"📁 {item} (no access)", item_path, "directory"))
            except PermissionError:
                print("❌ Permission denied to read this directory")
                
            # Show current directory option
            csv_files_here = 0
            try:
                csv_files_here = len([f for f in os.listdir(current_path) 
                                    if f.lower().endswith(('.csv', '.xlsx', '.xls'))])
            except:
                pass
            
            print(f"0. 📍 SELECT THIS DIRECTORY ({csv_files_here} data files found)")
            print("-" * 60)
            
            # Display directories
            for i, (display_name, full_path, item_type) in enumerate(items, 1):
                print(f"{i}. {display_name}")
            
            print("-" * 60)
            print("📋 Other options:")
            print("h. Go to Home directory")
            print("d. Go to Desktop")
            print("doc. Go to Documents") 
            print("down. Go to Downloads")
            print("c. Enter custom path")
            print("q. Quit and use current directory")
            
            # Get user choice
            choice = input(f"\nSelect option (0-{len(items)}, h, d, doc, down, c, q): ").strip().lower()
            
            if choice == "0":
                return current_path
            elif choice == "q":
                return "."
            elif choice == "h":
                current_path = os.path.expanduser("~")
            elif choice == "d":
                desktop_path = os.path.expanduser("~/Desktop")
                if os.path.exists(desktop_path):
                    current_path = desktop_path
                else:
                    print("❌ Desktop directory not found")
            elif choice == "doc":
                docs_path = os.path.expanduser("~/Documents")
                if os.path.exists(docs_path):
                    current_path = docs_path
                else:
                    print("❌ Documents directory not found")
            elif choice == "down":
                downloads_path = os.path.expanduser("~/Downloads")
                if os.path.exists(downloads_path):
                    current_path = downloads_path
                else:
                    print("❌ Downloads directory not found")
            elif choice == "c":
                custom_path = input("Enter full path: ").strip()
                if custom_path.startswith('"') and custom_path.endswith('"'):
                    custom_path = custom_path[1:-1]
                if os.path.exists(custom_path) and os.path.isdir(custom_path):
                    current_path = os.path.abspath(custom_path)
                else:
                    print("❌ Invalid path or directory doesn't exist")
            else:
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(items):
                        selected_path = items[choice_idx][1]
                        if os.path.exists(selected_path):
                            current_path = selected_path
                        else:
                            print("❌ Directory no longer exists")
                    else:
                        print(f"❌ Please enter a number between 0 and {len(items)}")
                except ValueError:
                    print("❌ Invalid input. Please try again.")
                    
        except Exception as e:
            print(f"❌ Error browsing directory: {e}")
            return current_path

def show_folder_contents_preview(folder_path):
    """Show a preview of data files in the selected folder"""
    try:
        files = []
        for ext in ['*.csv', '*.xlsx', '*.xls']:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if files:
            print(f"\n📊 Data files found in {folder_path}:")
            print("-" * 50)
            for i, file in enumerate(files[:10], 1):  # Show first 10 files
                filename = os.path.basename(file)
                size = os.path.getsize(file) / 1024 / 1024  # Size in MB
                print(f"  {i}. {filename} ({size:.1f} MB)")
            
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
            
            print(f"\n✅ Total: {len(files)} data files")
        else:
            print(f"\n⚠️  No CSV or Excel files found in {folder_path}")
            
        return len(files) > 0
        
    except Exception as e:
        print(f"❌ Error reading folder contents: {e}")
        return False

def select_data_folder():
    """Enhanced folder selection with browsable directory options"""
    print("\nSELECT DATA FOLDER:")
    print("1. Use current directory")
    print("2. Browse and select from available directories")
    print("3. Enter custom folder path")
    print("4. Use common data folders")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        return "."
    
    elif choice == "2":
        return browse_directories()
    
    elif choice == "3":
        folder_path = input("Enter full folder path: ").strip()
        if folder_path.startswith('"') and folder_path.endswith('"'):
            folder_path = folder_path[1:-1]  # Remove quotes
        return folder_path
    
    elif choice == "4":
        common_folders = [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"), 
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/OneDrive/Desktop"),
            "C:\\Data",
            "."
        ]
        print("\nCommon folders:")
        for i, folder in enumerate(common_folders, 1):
            exists = "✓" if os.path.exists(folder) else "✗"
            print(f"{i}. {folder} {exists}")
        
        try:
            folder_choice = int(input(f"\nSelect folder (1-{len(common_folders)}): "))
            if 1 <= folder_choice <= len(common_folders):
                return common_folders[folder_choice - 1]
            else:
                return "."
        except:
            return "."
    else:
        return "."

def ask_continuous_futures_preference():
    """Ask user if they want to include continuous futures calculation"""
    print("\n" + "="*60)
    print("CONTINUOUS FUTURES CONTRACT OPTION")
    print("="*60)
    print("Would you like to include a continuous futures contract calculation?")
    print("This creates a synthetic constant maturity contract by interpolating")
    print("between front and next month contracts.")
    print("")
    print("Benefits:")
    print("- Eliminates rollover gaps and jumps")
    print("- Provides smooth price series")
    print("- Useful for backtesting and analysis")
    print("")
    
    while True:
        response = input("Include continuous futures? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

def get_target_days_to_expiry():
    """Get target days to expiry for continuous contract"""
    print("\nCONTINUOUS CONTRACT TARGET MATURITY")
    print("-" * 40)
    print("Enter target days to expiry for the synthetic contract.")
    print("Common values:")
    print("- 30 days (most common)")
    print("- 45 days") 
    print("- 60 days")
    print("")
    
    while True:
        try:
            days_input = input("Target days to expiry (default 30): ").strip()
            if not days_input:
                return 30
            days = int(days_input)
            if days > 0 and days <= 365:
                return days
            else:
                print("Please enter a number between 1 and 365")
        except ValueError:
            print("Please enter a valid number")

def main():
    """Main function to process futures data with STRICT Eastern Time rollover timing + Continuous Futures + ENHANCED MISSING DATA RECOVERY"""
    print("="*80)
    print("ENHANCED FUTURES DATA PROCESSOR - WITH CONTINUOUS CONTRACTS + MISSING DATA FIXES")
    print("="*80)
    print("All expiration times converted to Eastern Time")
    print("🔒 STRICT ROLLOVER: Contracts roll AT the exact expiration time (>=)")
    print("🔄 NEW: Continuous futures contract calculation available")
    print("🛠️ ENHANCED: Missing next month and price data recovery")
    print("CME futures: 9:30 AM CT → 10:30 AM ET")
    print("CBOE VIX futures: 9:00 AM ET (HARDCODED - no timezone conversion)")
    print("NYMEX/COMEX: Various ET times (no change)")
    print("")
    
    # Initialize the expiration calculator
    calc = FuturesExpirationCalculator()
    
    # Ask about continuous futures
    include_continuous = ask_continuous_futures_preference()
    continuous_calc = None
    target_days = 30
    
    if include_continuous:
        target_days = get_target_days_to_expiry()
        continuous_calc = ContinuousFuturesCalculator(target_days)
        print(f"\n✓ Continuous futures calculation enabled (target: {target_days} days)")
    
    # Enhanced interactive folder selection
    data_folder = select_data_folder()
    
    print(f"\n✓ Selected folder: {data_folder}")
    
    # Show preview of files in selected folder
    has_files = show_folder_contents_preview(data_folder)
    
    if not has_files:
        print("\n❌ No data files found. Would you like to:")
        print("1. Select a different folder")
        print("2. Continue anyway")
        print("3. Exit")
        
        choice = input("Select option (1-3): ").strip()
        if choice == "1":
            data_folder = select_data_folder()
            has_files = show_folder_contents_preview(data_folder)
        elif choice == "3":
            return
    
    # Find all data files in selected folder
    files = glob.glob(os.path.join(data_folder, "*.csv")) + glob.glob(os.path.join(data_folder, "*.xlsx"))
    
    if not files:
        print(f"\n✗ No CSV or Excel files found in: {data_folder}")
        print("\nTroubleshooting:")
        print("1. Make sure the folder contains .csv or .xlsx files")
        print("2. Check if files have the correct extensions")
        print("3. Try selecting a different folder")
        return
    
    # STEP 1: SELECT SPOT FILE
    print("\nSTEP 1: SELECT SPOT DATA FILE")
    print("-" * 30)
    for i, f in enumerate(files, 1):
        size = os.path.getsize(os.path.join(data_folder, f)) / 1024 / 1024
        print(f"{i}. {f} ({size:.1f} MB)")
    
    while True:
        try:
            spot_file_input = input("\nSelect spot file number: ").strip()
            spot_file_idx = int(spot_file_input) - 1
            if 0 <= spot_file_idx < len(files):
                break
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number")
    
    spot_file = os.path.join(data_folder, files[spot_file_idx])
    
    # Read spot file with robust encoding handling
    print("\nReading spot file...")
    spot_df = None
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            print(f"  Trying encoding: {encoding}")
            spot_df = pd.read_csv(spot_file, thousands=',', encoding=encoding)
            print(f"  ✓ Success with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"  ✗ Failed with {encoding}")
            continue
        except Exception as e:
            print(f"  ✗ Error with {encoding}: {e}")
            continue
    
    if spot_df is None:
        # Try Excel format
        try:
            print("  Trying Excel format...")
            spot_df = pd.read_excel(spot_file)
            print("  ✓ Success with Excel format")
        except Exception as e:
            print(f"Error reading file in any format: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. Open the file in Excel and save as CSV (UTF-8)")
            print("2. Check if file has special characters or unusual formatting")
            print("3. Ensure file is not corrupted or locked")
            return
    
    spot_cols = list(spot_df.columns)
    
    # Show first few rows for verification
    print(f"\n✓ Spot file loaded: {len(spot_df)} rows, {len(spot_df.columns)} columns")
    print("First few rows of spot data:")
    print(spot_df.head(3))
    print(f"\nSpot columns: {list(spot_df.columns)}")
    
    # Find date and close columns
    date_col = spot_cols[0]
    close_col = None
    
    # Look for close price column
    for col in spot_cols:
        if 'close' in col.lower():
            close_col = col
            break
    
    if not close_col:
        # Look for any price column
        for col in spot_cols[1:]:
            if any(term in col.upper() for term in ['PRICE', 'LAST', 'CLOSE', 'VIX', 'SPX', 'IWM']):
                close_col = col
                break
    
    if not close_col:
        close_col = spot_cols[1]  # Default to second column
    
    print(f"✓ Using {close_col} as spot close price")
    
    # STEP 2: SELECT FUTURES FILE
    print("\nSTEP 2: SELECT FUTURES DATA FILE")
    print("-" * 30)
    for i, f in enumerate(files, 1):
        size = os.path.getsize(os.path.join(data_folder, f)) / 1024 / 1024
        print(f"{i}. {f} ({size:.1f} MB)")
    
    while True:
        try:
            futures_file_input = input("\nSelect futures file number: ").strip()
            futures_file_idx = int(futures_file_input) - 1
            if 0 <= futures_file_idx < len(files):
                break
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number")
    
    futures_file = os.path.join(data_folder, files[futures_file_idx])
    
    # Read futures file with robust encoding handling
    print("\nReading futures file...")
    futures_df = None
    
    for encoding in encodings_to_try:
        try:
            print(f"  Trying encoding: {encoding}")
            futures_df = pd.read_csv(futures_file, thousands=',', encoding=encoding)
            print(f"  ✓ Success with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"  ✗ Failed with {encoding}")
            continue
        except Exception as e:
            print(f"  ✗ Error with {encoding}: {e}")
            continue
    
    if futures_df is None:
        # Try Excel format
        try:
            print("  Trying Excel format...")
            futures_df = pd.read_excel(futures_file)
            print("  ✓ Success with Excel format")
        except Exception as e:
            print(f"Error reading futures file: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. Open the file in Excel and save as CSV (UTF-8)")
            print("2. Check if file has special characters or unusual formatting")
            print("3. Ensure file is not corrupted or locked")
            return
    
    futures_cols = list(futures_df.columns)
    futures_date_col = futures_cols[0]
    
    # Get close price columns
    contract_cols = []
    contract_info = {}  # Maps column to (futures_type, contract_id)
    
    for col in futures_cols[1:]:
        if 'close' in col.lower():
            futures_type, contract_id = extract_contract_info(col)
            if futures_type and contract_id:
                contract_cols.append(col)
                contract_info[col] = (futures_type, contract_id)
    
    if not contract_cols:
        print("⚠️  No close price columns found, using all non-date columns")
        for col in futures_cols[1:]:
            futures_type, contract_id = extract_contract_info(col)
            if futures_type and contract_id:
                contract_cols.append(col)
                contract_info[col] = (futures_type, contract_id)
    
    # Detect futures type
    main_futures_type = detect_futures_type(contract_cols)
    
    print(f"✓ Detected {main_futures_type} futures")
    print(f"✓ Found {len(contract_cols)} futures contracts")
    
    # =======================================================================
    # DIAGNOSTIC SECTION - CONTRACT DETECTION - FIXED TIMEZONE ISSUES
    # =======================================================================
    print("\n" + "="*60)
    print("CONTRACT DETECTION DIAGNOSTIC")
    print("="*60)
    
    # Show all detected contracts
    print(f"All contract columns found ({len(contract_cols)}):")
    for i, col in enumerate(contract_cols[:15]):  # Show first 15
        if col in contract_info:
            futures_type, contract_id = contract_info[col]
            expiry = calc.calculate_expiration(contract_id, futures_type)
            if expiry:
                print(f"  {i+1:2d}. {col:<35} -> {contract_id:<8} (expires {expiry.strftime('%Y-%m-%d %H:%M')})")
    
    print(f"\nChecking first few rows of data for active contracts...")
    
    # Check the first few rows to see what contracts have data
    sample_rows = futures_df.head(3)
    print(f"\nSample from first 3 rows:")
    
    for idx, row in sample_rows.iterrows():
        current_date = row[futures_date_col]
        print(f"\nRow {idx+1} - Date: {current_date}")
        
        active_count = 0
        for col in contract_cols[:10]:  # Check first 10 contracts
            if col in contract_info:
                futures_type, contract_id = contract_info[col]
                expiry = calc.calculate_expiration(contract_id, futures_type)
                
                if expiry:
                    # Use STRICT expiration check
                    has_expired = has_contract_expired_strict(expiry, current_date)
                    
                    if not has_expired:  # Contract is still active
                        # Use enhanced price finder
                        price = find_contract_price_enhanced(
                            contract_id, row, contract_cols, contract_info, futures_df, current_date
                        )
                        
                        if price and price > 0:
                            is_active, time_diff = safe_datetime_compare(expiry, current_date)
                            days_to_exp = time_diff.total_seconds() / 86400
                            print(f"    {contract_id}: ${price:.2f} ({days_to_exp:.1f} days to expiry)")
                            active_count += 1
        
        print(f"    Total active contracts: {active_count}")
    
    print("\n" + "="*60)
    
    # Also check what happens during initialization
    print("TESTING INITIALIZATION LOGIC...")
    
    # Simulate the initialization process with proper timezone handling
    first_date_raw = futures_df[futures_date_col].iloc[0]
    test_date = standardize_datetime_to_eastern(pd.Series([first_date_raw]), main_futures_type).iloc[0]
    date_rows = futures_df[futures_df[futures_date_col] == first_date_raw]
    
    print(f"Test date: {test_date}")
    print(f"Rows for this date: {len(date_rows)}")
    
    # Get all active contracts for this test date
    active_contracts = []
    for col in contract_cols:
        if col in contract_info:
            futures_type, contract_id = contract_info[col]
            expiry = calc.calculate_expiration(contract_id, futures_type)
            
            if expiry:
                # Use STRICT expiration check
                has_expired = has_contract_expired_strict(expiry, test_date)
                
                if not has_expired:  # Contract is still active
                    for _, row in date_rows.iterrows():
                        # Use enhanced price finder
                        price = find_contract_price_enhanced(
                            contract_id, row, contract_cols, contract_info, futures_df, test_date
                        )
                        
                        if price and price > 0:
                            is_active, time_diff = safe_datetime_compare(expiry, test_date)
                            days_to_exp = time_diff.total_seconds() / 86400
                            
                            active_contracts.append({
                                'col': col,
                                'contract_id': contract_id,
                                'expiry': expiry,
                                'price': price,
                                'days': days_to_exp
                            })
                            break  # Found price for this contract on this date
    
    print(f"\nActive contracts found for initialization: {len(active_contracts)}")
    if len(active_contracts) >= 2:
        active_contracts.sort(key=lambda x: x['expiry'])
        print("✅ FOUND MULTIPLE CONTRACTS")
        print(f"  Front: {active_contracts[0]['contract_id']} (${active_contracts[0]['price']:.2f})")
        print(f"  Next:  {active_contracts[1]['contract_id']} (${active_contracts[1]['price']:.2f})")
    elif len(active_contracts) == 1:
        print("⚠️  ONLY ONE CONTRACT HAS PRICE DATA")
        print(f"  Only: {active_contracts[0]['contract_id']} (${active_contracts[0]['price']:.2f})")
        print("  Other contracts may start trading later...")
    else:
        print("❌ NO CONTRACTS FOUND - Check data format")
        print("   This means no contracts have valid price data on the first date")
    
    print("="*60)
    
    # Wait for user to review diagnostics
    input("\nPress Enter to continue with processing after reviewing diagnostics...")
    
    # =======================================================================
    # END DIAGNOSTIC SECTION
    # =======================================================================
    
    # Ask user if they want to verify expiration dates
    verify_dates = ask_verification_preference()  # BRING BACK EXPIRATION CHECK
    
    if verify_dates:
        # Interactive expiration verification with calendar display
        verified_contracts = verify_expiration_dates(calc, contract_cols, contract_info)
        update_calculator_with_verified_dates(calc, verified_contracts)
        print("\n✅ Using verified expiration dates for processing")
    else:
        print("\n⚠️ Using calculated expiration dates without verification")
    
    print("\n" + "="*60)
    print("PROCEEDING TO DATA PROCESSING WITH STRICT ROLLOVER + ENHANCED MISSING DATA RECOVERY...")
    if include_continuous:
        print(f"Including continuous futures calculation (target: {target_days} days)")
    print("="*60)
    
    # Show detected expiration dates (verified or calculated)
    print(f"\nContract Expiration Schedule (Eastern Time):")
    print("-" * 60)
    shown_contracts = set()
    for col in contract_cols[:10]:  # Show first 10
        if col in contract_info:
            futures_type, contract_id = contract_info[col]
            if contract_id not in shown_contracts:
                # Use our dynamic calculator (now with verified dates if applicable)
                expiry = calc.calculate_expiration(contract_id, futures_type)
                if expiry:
                    # Show original exchange time and ET conversion with DST awareness
                    spec = calc.futures_specs.get(futures_type, {})
                    native_tz = spec.get('native_timezone', 'US/Eastern')
                    native_time = spec.get('native_time', (0, 0))
                    
                    if native_tz == 'US/Central':
                        # Determine if this date is during DST
                        is_dst = expiry.dst() != timedelta(0)
                        tz_abbrev = "CDT" if is_dst else "CST"
                        verified_note = " ✅" if hasattr(calc, 'verified_expirations') and contract_id in calc.verified_expirations else ""
                        print(f"{contract_id}: {expiry.strftime('%B %d, %Y at %I:%M %p ET')} (from {native_time[0]}:{native_time[1]:02d} {tz_abbrev}){verified_note}")
                    else:
                        verified_note = " ✅" if hasattr(calc, 'verified_expirations') and contract_id in calc.verified_expirations else ""
                        print(f"{contract_id}: {expiry.strftime('%B %d, %Y at %I:%M %p ET')} (native ET){verified_note}")
                    shown_contracts.add(contract_id)
    print("-" * 60)
    
    # Parse dates and standardize to Eastern Time
    print("\nStandardizing timestamps to Eastern Time...")
    spot_df[date_col] = pd.to_datetime(spot_df[date_col])
    spot_df[date_col] = standardize_datetime_to_eastern(spot_df[date_col], main_futures_type)
    
    futures_df[futures_date_col] = pd.to_datetime(futures_df[futures_date_col])
    futures_df[futures_date_col] = standardize_datetime_to_eastern(futures_df[futures_date_col], main_futures_type)
    
    # Merge data - using outer merge to keep all rows
    print("Merging data...")
    merged_df = pd.merge(
        spot_df[[date_col, close_col]],
        futures_df,
        left_on=date_col,
        right_on=futures_date_col,
        how='outer'  # Keep all rows
    )
    
    # Create a unified date column
    merged_df['unified_date'] = merged_df[date_col].combine_first(merged_df[futures_date_col])
    
    print(f"✓ Merged {len(merged_df)} rows")
    
    # Process data
    print(f"\nProcessing contracts with STRICT Eastern Time roll detection...")
    if include_continuous:
        print(f"+ Calculating continuous futures contract ({target_days}-day target maturity)")
    
    results = []
    
    # Track current contracts and rolls
    current_front = None
    current_next = None
    contracts_initialized = False
    roll_dates = []
    
    # Initial tracking for active contracts
    all_contract_data = {}
    
    # First pass - identify front and next month contracts across all dates
    all_dates = sorted(merged_df['unified_date'].dropna().unique())
    
    # Initialize contracts on first date with sufficient data
    for current_date in all_dates:
        if contracts_initialized:
            break
            
        date_rows = merged_df[merged_df['unified_date'] == current_date]
        
        # Get all active contracts for this date using STRICT expiration check + ENHANCED PRICE FINDER
        active_contracts = []
        for col in contract_cols:
            if col in contract_info:
                futures_type, contract_id = contract_info[col]
                expiry = calc.calculate_expiration(contract_id, futures_type)
                
                if expiry:
                    # Use STRICT expiration check
                    has_expired = has_contract_expired_strict(expiry, current_date)
                    
                    if not has_expired:  # Contract is still active
                        for _, row in date_rows.iterrows():
                            # Use enhanced price finder
                            price = find_contract_price_enhanced(
                                contract_id, row, contract_cols, contract_info, merged_df, current_date
                            )
                            
                            if price and price > 0:
                                is_active, time_diff = safe_datetime_compare(expiry, current_date)
                                days_to_exp = time_diff.total_seconds() / 86400
                                
                                active_contracts.append({
                                    'col': col,
                                    'contract_id': contract_id,
                                    'expiry': expiry,
                                    'price': price,
                                    'days': days_to_exp
                                })
                                # Store in our tracking dictionary
                                all_contract_data[contract_id] = {
                                    'col': col,
                                    'expiry': expiry,
                                    'price': price,
                                    'days': days_to_exp
                                }
                                break  # Found price for this contract on this date
        
        # ENHANCED INITIALIZATION WITH ROBUST NEXT CONTRACT LOGIC
        if len(active_contracts) >= 1:
            # Sort by expiration
            active_contracts.sort(key=lambda x: x['expiry'])
            
            # ALWAYS use the earliest expiring contract as front month
            current_front = active_contracts[0]['contract_id']
            
            print(f"\nDEBUG: Starting with front month: {current_front}")
            print(f"DEBUG: All active contracts: {[c['contract_id'] for c in active_contracts]}")
            
            # Use enhanced next month finder
            current_next = find_next_month_contract_enhanced(
                current_front, active_contracts, current_date, calc, main_futures_type
            )
            
            contracts_initialized = True
            
            print(f"\n" + "="*60)
            print(f"FINAL INITIALIZATION WITH ENHANCED ROBUST LOGIC:")
            print(f"  Front Contract: {current_front}")
            print(f"  Next Contract: {current_next if current_next else 'None (Will use emergency search)'}")
            
            if current_next:
                front_expiry = calc.calculate_expiration(current_front, main_futures_type)
                next_expiry = calc.calculate_expiration(current_next, main_futures_type)
                print(f"  Front expires: {front_expiry.strftime('%B %d, %Y at %I:%M %p')}")
                print(f"  Next expires: {next_expiry.strftime('%B %d, %Y at %I:%M %p')}")
                print(f"  ✅ Sequence: {current_front} → {current_next}")
            else:
                print(f"  ⚠️ Will rely on enhanced contract search for each row")
            print("="*60)
    
    if not contracts_initialized:
        print("\n⚠️ Could not initialize contracts - not enough data!")
        return
    
    # Now process each row with STRICT rollover timing + ENHANCED DATA RECOVERY
    print(f"\nProcessing {len(merged_df)} rows of data with STRICT rollover logic + ENHANCED RECOVERY...")
    for idx, row in merged_df.iterrows():
        if idx % 25000 == 0:  # Show progress every 25k rows
            percent = (idx / len(merged_df)) * 100
            print(f"  Progress: {idx:,}/{len(merged_df):,} rows ({percent:.1f}%)")
        
        current_date = row['unified_date']
        if pd.isna(current_date):
            continue
            
        # Get spot close - might be NaN
        spot_close = row[close_col] if date_col in row and not pd.isna(row[date_col]) else None
        
        # Get all active contracts for this row using ENHANCED PRICE FINDER
        active_contracts = []
        for col in contract_cols:
            if col in contract_info:
                futures_type, contract_id = contract_info[col]
                expiry = calc.calculate_expiration(contract_id, futures_type)
                
                if expiry:
                    # Use STRICT expiration check
                    has_expired = has_contract_expired_strict(expiry, current_date)
                    
                    if not has_expired:  # Contract is still active
                        # Use enhanced price finder
                        price = find_contract_price_enhanced(
                            contract_id, row, contract_cols, contract_info, merged_df, current_date
                        )
                        
                        if price and price > 0:
                            is_active, time_diff = safe_datetime_compare(expiry, current_date)
                            days_to_exp = time_diff.total_seconds() / 86400
                            
                            active_contracts.append({
                                'col': col,
                                'contract_id': contract_id,
                                'expiry': expiry,
                                'price': price,
                                'days': days_to_exp
                            })
                            
                            # Update our tracking
                            all_contract_data[contract_id] = {
                                'col': col,
                                'expiry': expiry,
                                'price': price,
                                'days': days_to_exp
                            }
        
        # ENHANCED NEXT CONTRACT FINDING using the same month sequence logic
        if current_next is None and current_front:
            current_next = find_next_month_contract_enhanced(
                current_front, active_contracts, current_date, calc, main_futures_type
            )
            if current_next:
                print(f"\n🔄 ENHANCED SEARCH FOUND NEXT CONTRACT: {current_next}")
                print(f"   Updated sequence: {current_front} → {current_next}")
        
        # Check for roll - STRICT LOGIC: Roll immediately when expiry detected
        front_found = False
        roll_occurred = False
        front_expiry_check = calc.calculate_expiration(current_front, main_futures_type)
        
        if front_expiry_check:
            # Use STRICT expiration check
            has_expired = has_contract_expired_strict(front_expiry_check, current_date)
            
            # IMMEDIATE ROLLOVER: If contract has expired, roll immediately
            if has_expired:  # Expiration time has been reached
                print(f"\n🔒 STRICT ROLLOVER: {current_front} has EXPIRED at {current_date.strftime('%Y-%m-%d %H:%M')} (expiry was {front_expiry_check.strftime('%Y-%m-%d %H:%M')})")
                
                # ROLL IMMEDIATELY - don't wait for next timestamp
                old_front = current_front
                old_next = current_next
                
                print(f"\n📅 IMMEDIATE CONTRACT ROLL at {current_date.strftime('%Y-%m-%d %H:%M')}:")
                print(f"   {old_front} has expired, rolling now...")
                
                # Use enhanced next month finding logic for new front and next
                new_front = None
                new_next = None
                
                # METHOD 1: Try to use the current next month as new front
                if current_next:
                    # Check if current next month hasn't expired using STRICT check
                    next_expiry = calc.calculate_expiration(current_next, main_futures_type)
                    if next_expiry:
                        has_expired_next = has_contract_expired_strict(next_expiry, current_date)
                        if not has_expired_next:
                            new_front = current_next
                            print(f"   New front month: {new_front} (was next month)")
                            
                            # Find the new next month using enhanced logic
                            new_next = find_next_month_contract_enhanced(
                                new_front, active_contracts, current_date, calc, main_futures_type
                            )
                            if new_next:
                                print(f"   New next month: {new_next}")
                
                # METHOD 2: Fallback - Find any active contract as new front
                if not new_front:
                    print(f"   METHOD 2: Finding any active contract as new front")
                    active_contracts.sort(key=lambda x: x['expiry'])
                    for contract in active_contracts:
                        has_expired_contract = has_contract_expired_strict(contract['expiry'], current_date)
                        if not has_expired_contract and contract['contract_id'] != old_front:
                            new_front = contract['contract_id']
                            print(f"   Fallback new front: {new_front}")
                            
                            # Find next contract using enhanced logic
                            new_next = find_next_month_contract_enhanced(
                                new_front, active_contracts, current_date, calc, main_futures_type
                            )
                            if new_next:
                                print(f"   Fallback new next: {new_next}")
                            break
                
                if new_front and new_front != old_front:
                    current_front = new_front
                    current_next = new_next
                    roll_occurred = True
                    roll_dates.append(current_date)
                    
                    print(f"   ✅ IMMEDIATE ROLL COMPLETED:")
                    print(f"   Front: {old_front} → {current_front}")
                    if current_next:
                        print(f"   Next: {old_next} → {current_next}")
                    else:
                        print(f"   Next: {old_next} → None (no next contract available)")
                
                front_found = False  # Contract has rolled
            else:
                # Contract is still active - look for it in current data
                for contract in active_contracts:
                    if contract['contract_id'] == current_front:
                        front_found = True
                        break
                # If we can't find the front contract in data but it hasn't expired, keep using it
                if not front_found:
                    front_found = True  # Keep current contract until it actually expires
        
        # Get contract data using ENHANCED ROBUST SYSTEM
        front_data = get_contract_data_robust(
            current_front, row, contract_cols, contract_info, calc, 
            main_futures_type, current_date, merged_df
        )
        
        if current_next:
            next_data = get_contract_data_robust(
                current_next, row, contract_cols, contract_info, calc, 
                main_futures_type, current_date, merged_df
            )
        else:
            next_data = None
        
        # Enhanced debugging for critical moments
        if (current_date.strftime('%H:%M') in ['09:00', '09:01'] or 
            roll_occurred or idx % 10000 == 0):
            
            print(f"\n📊 ENHANCED STATUS CHECK: {current_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Front: {current_front} -> Price: {front_data['price'] if front_data else 'None'}")
            print(f"   Next:  {current_next} -> Price: {next_data['price'] if next_data else 'None'}")
            print(f"   Active contracts: {len(active_contracts)}")
            
            if not front_data or front_data['price'] is None:
                print(f"   🔍 DIAGNOSING MISSING FRONT PRICE:")
                # Show what's available in this row
                for col in contract_cols[:5]:
                    try:
                        val = row[col]
                        if not pd.isna(val) and val != 0:
                            print(f"      {col}: {val}")
                    except:
                        pass
        
        # Calculate continuous futures if enabled - ENHANCED with aggressive fallback
        continuous_price = None
        front_weight = None
        next_weight = None
        
        if include_continuous and continuous_calc:
            # STEP 1: Enhanced front month price retrieval
            front_price = None
            front_days = None
            
            # Method 1: Use front_data if available
            if front_data and front_data.get('price') is not None:
                front_price = front_data['price']
                front_days = front_data.get('days')
            
            # Method 2: Enhanced fallback - find ANY valid price in this row
            if front_price is None:
                # Use our enhanced price finder directly
                front_price = find_contract_price_enhanced(
                    current_front, row, contract_cols, contract_info, merged_df, current_date
                )
                if front_price:
                    # Calculate days to expiry
                    expiry = calc.calculate_expiration(current_front, main_futures_type)
                    if expiry:
                        is_active, time_diff = safe_datetime_compare(expiry, current_date)
                        front_days = time_diff.total_seconds() / 86400
            
            # Method 3: Emergency fallback - find ANY valid price in this row
            if front_price is None:
                for col in contract_cols:
                    if col in contract_info:
                        futures_type, contract_id = contract_info[col]
                        # Use enhanced price finder
                        test_price = find_contract_price_enhanced(
                            contract_id, row, contract_cols, contract_info, merged_df, current_date
                        )
                        if test_price and test_price > 0:
                            expiry = calc.calculate_expiration(contract_id, futures_type)
                            if expiry:
                                has_expired = has_contract_expired_strict(expiry, current_date)
                                if not has_expired:  # Contract is still valid
                                    front_price = test_price
                                    is_active, time_diff = safe_datetime_compare(expiry, current_date)
                                    front_days = time_diff.total_seconds() / 86400
                                    # Log emergency use
                                    if (current_date.strftime('%H:%M') == '09:00' or 
                                        current_date.strftime('%H:%M') == '09:01' or 
                                        roll_occurred or idx % 5000 == 0):
                                        print(f"   📊 Emergency: Using {contract_id} price ${front_price:.2f} for continuous at {current_date.strftime('%Y-%m-%d %H:%M')}")
                                    break
            
            # STEP 2: Enhanced next month data
            next_price = None
            next_days = None
            
            if next_data and next_data.get('price') is not None:
                next_price = next_data['price']
                next_days = next_data.get('days')
            elif current_next:
                # Use enhanced price finder
                next_price = find_contract_price_enhanced(
                    current_next, row, contract_cols, contract_info, merged_df, current_date
                )
                if next_price:
                    expiry = calc.calculate_expiration(current_next, main_futures_type)
                    if expiry:
                        is_active, time_diff = safe_datetime_compare(expiry, current_date)
                        next_days = time_diff.total_seconds() / 86400
            
            # STEP 3: Calculate continuous price with enhanced debugging
            if front_price is not None:
                if next_price is not None and next_days is not None and front_days is not None:
                    # Normal case: both front and next month available
                    continuous_price = continuous_calc.calculate_continuous_price(
                        front_price, 
                        next_price, 
                        front_days, 
                        next_days
                    )
                    front_weight, next_weight = continuous_calc.calculate_weights(
                        front_days, 
                        next_days
                    )
                    # Debug rollover specifically
                    if current_date.strftime('%H:%M') in ['09:00', '09:01'] or roll_occurred:
                        print(f"   📊 INTERPOLATED CONTINUOUS: {current_date.strftime('%Y-%m-%d %H:%M')} = ${continuous_price:.2f}")
                        print(f"       Front: ${front_price:.2f} ({front_days:.1f}d) * {front_weight:.3f}")
                        print(f"       Next:  ${next_price:.2f} ({next_days:.1f}d) * {next_weight:.3f}")
                else:
                    # Fallback: Use front month only when next month unavailable
                    continuous_price = front_price
                    front_weight = 1.0
                    next_weight = 0.0
                    # Debug rollover specifically
                    if (current_date.strftime('%H:%M') in ['09:00', '09:01'] or roll_occurred):
                        print(f"   📊 FRONT-ONLY CONTINUOUS: {current_date.strftime('%Y-%m-%d %H:%M')} = ${front_price:.2f} (front only)")
            else:
                # This should be much rarer now with enhanced price finding
                if current_date.strftime('%H:%M') in ['09:00', '09:01'] or roll_occurred:
                    print(f"   ❌ CRITICAL: No front price available at {current_date.strftime('%Y-%m-%d %H:%M')} for continuous calculation")
                    print(f"   Current front: {current_front}, Current next: {current_next}")
                    print(f"   Front data: {front_data}")
                    print(f"   Next data: {next_data}")
        
        # Get actual ticker symbols from the data
        spot_ticker, futures_ticker = detect_tickers_from_data(close_col, main_futures_type)
        
        # Prepare row data with dual contract tracking using actual tickers
        rollover_flag = "ROLL" if roll_occurred else ""
        
        # Create row with ticker-based column names
        row_data = {
            'Datetime': format_datetime_consistent(current_date),
            f'{spot_ticker}_Close': float(spot_close) if spot_close is not None else None,
            f'{futures_ticker}_Front_Month': current_front,
            f'{futures_ticker}_Front_Month_Close': front_data['price'] if front_data else None,
            f'{futures_ticker}_Next_Month': current_next,
            f'{futures_ticker}_Next_Month_Close': next_data['price'] if next_data else None,
            f'{futures_ticker}_Days_To_Front_Expiry': round(front_data['days'], 8) if front_data else None,
            f'{futures_ticker}_Days_To_Next_Expiry': round(next_data['days'], 8) if next_data else None,
            'Rollover': rollover_flag
        }
        
        # Add continuous futures columns if enabled
        if include_continuous:
            row_data[f'{futures_ticker}_Continuous_{target_days}D'] = continuous_price
            row_data[f'{futures_ticker}_Front_Weight'] = round(front_weight, 6) if front_weight is not None else None
            row_data[f'{futures_ticker}_Next_Weight'] = round(next_weight, 6) if next_weight is not None else None
        
        results.append(row_data)
    
    # Create output
    output_df = pd.DataFrame(results)
    print(f"\n✓ Processed {len(output_df)} rows with STRICT rollover timing + ENHANCED RECOVERY")
    if include_continuous:
        print(f"✓ Continuous futures calculated with {target_days}-day target maturity")
    
    # Save as CSV with ticker-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spot_ticker, futures_ticker = detect_tickers_from_data(close_col, main_futures_type)
    
    continuous_suffix = f"_CONTINUOUS_{target_days}D" if include_continuous else ""
    output_file = f"{spot_ticker}_{futures_ticker}_Futures_STRICT_ET{continuous_suffix}_ENHANCED_Processed_{timestamp}.csv"
    
    try:
        output_df.to_csv(output_file, index=False)
        print(f"\n✓ SAVED: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving: {e}")
        alt_file = os.path.expanduser(f"~/{spot_ticker}_{futures_ticker}_Futures_STRICT_ET{continuous_suffix}_ENHANCED_Processed_{timestamp}.csv")
        try:
            output_df.to_csv(alt_file, index=False)
            print(f"✓ Saved to: {alt_file}")
        except:
            print("✗ Could not save file")
    
    # Summary with timezone info
    print(f"\nSUMMARY - ENHANCED FUTURES PROCESSOR WITH MISSING DATA RECOVERY:")
    print("-" * 70)
    print(f"- Futures Type: {main_futures_type}")
    print(f"- Total rows: {len(output_df):,}")
    print(f"- Date range: {output_df['Datetime'].iloc[0]} to {output_df['Datetime'].iloc[-1]}")
    print(f"- Contract rolls: {len(roll_dates)}")
    
    if include_continuous:
        print(f"- Continuous contract: {target_days}-day target maturity")
        continuous_col = f'{futures_ticker}_Continuous_{target_days}D'
        if continuous_col in output_df.columns:
            continuous_data = output_df[continuous_col].dropna()
            if len(continuous_data) > 0:
                print(f"- Continuous price range: ${continuous_data.min():.2f} to ${continuous_data.max():.2f}")
    
    if roll_dates:
        print(f"\nContract Roll Dates (STRICT TIMING):")
        for i, roll_date in enumerate(roll_dates[:5]):
            print(f"  {i+1}. {roll_date.strftime('%Y-%m-%d %H:%M')} ET")
        if len(roll_dates) > 5:
            print(f"  ... and {len(roll_dates) - 5} more rolls")
    
    print(f"\n🔒 STRICT ROLLOVER + ENHANCED MISSING DATA RECOVERY APPLIED")
    print(f"✓ Contracts roll AT the exact expiration time")
    print(f"✓ Enhanced next month contract detection with multiple fallback strategies")
    print(f"✓ Enhanced price data recovery with time-window fallback")
    print(f"✓ Front and next month contract tracking enabled")
    if include_continuous:
        print(f"✓ Continuous {target_days}-day futures contract calculated")
        print(f"✓ Interpolation weights provided for transparency")
    print(f"✓ All timestamps standardized to Eastern Time")
    print(f"✓ Enhanced price data recovery with time-window fallback")
    print(f"✓ Front and next month contract tracking enabled")
    print(f"✓ Continuous futures contract calculation available")
    print(f"✓ Interpolation weights provided for transparency")
    print(f"✓ All timestamps standardized to Eastern Time")
    print(f"✓ Missing data boxes should now be filled!")

if __name__ == "__main__":
    # Run main processor with STRICT rollover timing, continuous futures, and enhanced missing data recovery
    main()