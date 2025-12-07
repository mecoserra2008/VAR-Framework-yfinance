# ============================================================================
# data_loader.py - Data Loading and Transformation Module
# ============================================================================
"""
This module handles:
- Loading assets from YAML configuration
- Fetching financial data from Yahoo Finance  
- Data transformations (log, differences, returns)
"""

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# YAML LOADER
# ============================================================================

@st.cache_data
def load_assets_from_yaml():
    """Load all assets from the YAML configuration file"""
    try:
        yaml_path = Path(r"C:\Users\User\Desktop\macro2\assets.yaml")
        if not yaml_path.exists():
            st.error("assets.yaml file not found. Please ensure it's in the same directory.")
            return {}
        
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Extract all assets into a flat dictionary
        all_assets = {}
        
        if 'asset_categories' in config:
            for category_key, category_data in config['asset_categories'].items():
                if 'symbols' in category_data:
                    for asset in category_data['symbols']:
                        symbol = asset.get('symbol')
                        if symbol:
                            all_assets[symbol] = {
                                'name': asset.get('name', symbol),
                                'category': asset.get('category', 'Unknown'),
                                'description': asset.get('description', ''),
                                'category_group': category_data.get('name', category_key)
                            }
        
        return all_assets
    except Exception as e:
        st.error(f"Error loading assets from YAML: {e}")
        return {}

# Treasury yields - add to available assets
TREASURY_YIELDS = {
    "^IRX": {"name": "13 Week Treasury Bill", "category": "Treasury_Yield", "description": "3-Month T-Bill Yield", "category_group": "US Treasuries"},
    "^FVX": {"name": "5-Year Treasury Yield", "category": "Treasury_Yield", "description": "5-Year Treasury Yield", "category_group": "US Treasuries"},
    "^TNX": {"name": "10-Year Treasury Yield", "category": "Treasury_Yield", "description": "10-Year Treasury Yield", "category_group": "US Treasuries"},
    "^TYX": {"name": "30-Year Treasury Yield", "category": "Treasury_Yield", "description": "30-Year Treasury Yield", "category_group": "US Treasuries"},
}

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_data(symbols, names, start_date, end_date, frequency='daily'):
    """Fetch data for selected symbols with batch retrieval over time periods"""
    try:
        data_dict = {}
        failed_symbols = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Calculate total days and determine batch size
        total_days = (end_dt - start_dt).days

        # Use batch retrieval for periods longer than 2 years
        use_batch = total_days > 730

        if use_batch:
            # Split into 1-year batches
            batch_size_days = 365
            st.info(f"📦 Using batch retrieval: splitting {total_days} days into chunks of ~{batch_size_days} days")

        for idx, (symbol, name) in enumerate(zip(symbols, names)):
            try:
                status_text.text(f"Fetching {name} ({symbol})... ({idx+1}/{len(symbols)})")

                if use_batch:
                    # Batch retrieval
                    all_data = []
                    current_start = start_dt
                    batch_num = 1

                    while current_start < end_dt:
                        current_end = min(current_start + timedelta(days=batch_size_days), end_dt)

                        status_text.text(f"Fetching {name} ({symbol})... Batch {batch_num} [{current_start.date()} to {current_end.date()}]")

                        try:
                            batch_data = yf.download(
                                symbol,
                                start=current_start.strftime('%Y-%m-%d'),
                                end=current_end.strftime('%Y-%m-%d'),
                                progress=False,
                                auto_adjust=True
                            )

                            if batch_data is not None and not batch_data.empty:
                                all_data.append(batch_data)

                            batch_num += 1
                        except Exception as e:
                            st.warning(f"Warning: Batch {batch_num} failed for {name}: {str(e)}")

                        # Move to next batch (add 1 day to avoid overlap)
                        current_start = current_end + timedelta(days=1)

                    # Combine all batches
                    if all_data:
                        ticker_data = pd.concat(all_data)
                        # Remove duplicates (might occur at batch boundaries)
                        ticker_data = ticker_data[~ticker_data.index.duplicated(keep='first')]
                        ticker_data = ticker_data.sort_index()
                    else:
                        ticker_data = pd.DataFrame()

                else:
                    # Single retrieval for shorter periods
                    ticker_data = yf.download(
                        symbol,
                        start=start_dt.strftime('%Y-%m-%d'),
                        end=end_dt.strftime('%Y-%m-%d'),
                        progress=False,
                        auto_adjust=True
                    )

                # Check if data is valid
                if ticker_data is None or ticker_data.empty:
                    failed_symbols.append((symbol, name, "No data returned"))
                    continue

                # Handle multi-index columns (when yfinance returns multi-level columns)
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    # Flatten multi-index columns
                    ticker_data.columns = ticker_data.columns.get_level_values(0)

                # Extract close prices with multiple fallback options
                close_series = None
                
                # Try different column names
                for col_name in ['Close', 'close', 'Adj Close', 'adj close']:
                    if col_name in ticker_data.columns:
                        close_series = ticker_data[col_name]
                        break
                
                # If still no close column, try to use the first numeric column
                if close_series is None:
                    if len(ticker_data.columns) == 1:
                        close_series = ticker_data.iloc[:, 0]
                    elif len(ticker_data.columns) > 0:
                        # Try to find any price-related column
                        price_cols = [col for col in ticker_data.columns if any(x in str(col).lower() for x in ['close', 'price', 'adj'])]
                        if price_cols:
                            close_series = ticker_data[price_cols[0]]
                        else:
                            close_series = ticker_data.iloc[:, 0]
                
                if close_series is None:
                    failed_symbols.append((symbol, name, "No price column found"))
                    continue

                # Convert to Series if needed and clean
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.squeeze()
                
                if not isinstance(close_series, pd.Series):
                    failed_symbols.append((symbol, name, "Could not convert to Series"))
                    continue

                # Drop NaN values
                close_series = close_series.dropna()

                # Validate data
                if len(close_series) == 0:
                    failed_symbols.append((symbol, name, "All values are NaN"))
                    continue
                
                if close_series.isnull().all():
                    failed_symbols.append((symbol, name, "All values are null"))
                    continue

                # Convert to numeric, coercing errors
                close_series = pd.to_numeric(close_series, errors='coerce').dropna()
                
                if len(close_series) == 0:
                    failed_symbols.append((symbol, name, "No numeric values"))
                    continue

                # Store valid data
                data_dict[name] = close_series
                st.success(f"✓ Fetched {len(close_series)} data points for {name}")

            except Exception as e:
                failed_symbols.append((symbol, name, str(e)))
                st.warning(f"Error fetching {name} ({symbol}): {str(e)}")

            progress_bar.progress((idx + 1) / len(symbols))

        progress_bar.empty()
        status_text.empty()

        # Check if we have any valid data
        if not data_dict:
            st.error("Failed to fetch data for all symbols. Please check your selection and try different assets.")
            if failed_symbols:
                st.error("Failed symbols details:")
                for sym, name, reason in failed_symbols:
                    st.write(f"- {name} ({sym}): {reason}")
            return pd.DataFrame()

        # Create DataFrame from dictionary of Series
        df = pd.DataFrame(data_dict)

        # Check if DataFrame is valid
        if df.empty:
            st.error("No overlapping data between selected assets.")
            return pd.DataFrame()

        # Drop rows with any NaN values to ensure complete observations
        df_clean = df.dropna()

        if df_clean.empty:
            st.error("No complete observations after removing missing data. Try different date range or assets.")
            return pd.DataFrame()

        # Check minimum sample size
        if len(df_clean) < 30:
            st.warning(f"Warning: Only {len(df_clean)} complete observations. Consider expanding date range.")

        # Resample if needed
        if frequency == 'weekly':
            df_clean = df_clean.resample('W-FRI').last().dropna()
        elif frequency == 'monthly':
            df_clean = df_clean.resample('M').last().dropna()

        # Final validation
        if df_clean.empty:
            st.error("No data remaining after resampling. Try different frequency or date range.")
            return pd.DataFrame()

        if len(df_clean) < 20:
            st.error(f"Insufficient data ({len(df_clean)} observations). Need at least 20 observations.")
            return pd.DataFrame()

        # Report success
        st.success(f" Successfully fetched data: {len(df_clean)} observations across {len(df_clean.columns)} series")
        
        if failed_symbols:
            with st.expander(" See failed symbols"):
                for sym, name, reason in failed_symbols:
                    st.write(f"- {name} ({sym}): {reason}")

        return df_clean

    except Exception as e:
        st.error(f"Critical error in data fetching: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def determine_transformation(df, adf_results):
    """Determine appropriate transformation based on unit root tests"""
    transformations = {}
    recommendations = {}
    
    for result in adf_results:
        var = result['Variable']
        is_stationary = result['Stationary (5%)']
        
        if is_stationary:
            transformations[var] = 'level'
            recommendations[var] = "Stationary in levels - use levels"
        else:
            if (df[var] > 0).all():
                transformations[var] = 'log_diff'
                recommendations[var] = "Non-stationary - recommend log-differencing (returns)"
            else:
                transformations[var] = 'diff'
                recommendations[var] = "Non-stationary - recommend first differencing"
    
    return transformations, recommendations

def apply_transformation(df, transformations):
    """Apply specified transformations to data"""
    df_transformed = df.copy()
    transform_info = {}
    
    for col in df.columns:
        if col in transformations:
            trans_type = transformations[col]
            if trans_type == 'log':
                if (df[col] > 0).all():
                    df_transformed[col] = np.log(df[col])
                    transform_info[col] = 'Log'
                else:
                    transform_info[col] = 'Level (log not possible - negative values)'
            elif trans_type == 'diff':
                df_transformed[col] = df[col].diff()
                transform_info[col] = 'First Difference'
            elif trans_type == 'log_diff':
                if (df[col] > 0).all():
                    df_transformed[col] = np.log(df[col]).diff()
                    transform_info[col] = 'Log Returns'
                else:
                    df_transformed[col] = df[col].diff()
                    transform_info[col] = 'First Difference (log not possible)'
            else:
                transform_info[col] = 'Level'
        else:
            transform_info[col] = 'Level'
    
    return df_transformed.dropna(), transform_info
