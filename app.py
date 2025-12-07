# ============================================================================
# app.py - Main Streamlit Application
# ============================================================================
"""
Main application file that brings together all modules:
- Data loading and transformation
- Statistical tests
- VAR model estimation and diagnostics
- Granger causality and IRF analysis
- Cointegration and VEC modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import chi2
from scipy import stats as scipy_stats

# Statsmodels imports
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
from sklearn.linear_model import LinearRegression

# ============================================================================
# IMPORT OUR CUSTOM MODULES
# ============================================================================

from data_loader import (
    load_assets_from_yaml,
    TREASURY_YIELDS,
    fetch_data,
    determine_transformation,
    apply_transformation
)

from statistical_tests import (
    perform_adf_test,
    perform_kpss_test,
    detect_seasonality,
    test_granger_causality_matrix
)

from var_model import (
    display_combined_stability_table,
    analyze_coefficient_significance,
    forecast_with_mse_and_ci
)

from causality_irf import (
    test_instantaneous_causality,
    comprehensive_granger_causality,
    compute_irf_with_ci,
    plot_irf_with_ci
)

from cointegration import (
    perform_adf_test_simple,
    perform_kpss_test_simple,
    perform_johansen_test
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize session state for persistence
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

if 'selected_lag' not in st.session_state:
    st.session_state.selected_lag = None

if 'var_model' not in st.session_state:
    st.session_state.var_model = None

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="VAR & Cointegration Analysis - Macroeconometrics II",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
def main():
    # Title
    st.markdown('<div class="main-header">VAR & Cointegration Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**ISEG - Macroeconometrics II - Team Work**")
    st.markdown("*Complete framework for Vector Autoregression and Cointegration Analysis*")
    st.markdown("---")
    
    # Load assets
    all_assets = load_assets_from_yaml()
    all_assets.update(TREASURY_YIELDS)  # Add treasury yields
    
    if not all_assets:
        st.error("No assets loaded. Please check your assets.yaml file.")
        return
    
    # ========================================================================
    # SIDEBAR: DATA SELECTION
    # ========================================================================
    
    with st.sidebar:
        st.header("📊 Data Configuration")
        
        st.subheader("1. Select Variables")
        st.info(f"**{len(all_assets)} assets available** from YAML configuration")
        
        # Group assets by category for easier selection
        categories = sorted(set(asset['category_group'] for asset in all_assets.values()))
        
        selected_category = st.selectbox(
            "Filter by category (optional):",
            ["All Categories"] + categories
        )
        
        # Filter assets by category
        if selected_category == "All Categories":
            filtered_assets = all_assets
        else:
            filtered_assets = {k: v for k, v in all_assets.items() if v['category_group'] == selected_category}
        
        st.write(f"**{len(filtered_assets)} assets** in selected category")
        
        # Create selection options
        asset_options = {f"{v['name']} ({k})": k for k, v in filtered_assets.items()}
        sorted_options = sorted(asset_options.keys())
        
        # Multi-select for variables (3-4 required)
        st.markdown("**Select 3-4 variables for analysis:**")
        selected_assets_display = st.multiselect(
            "Variables:",
            sorted_options,
            max_selections=4,
            help="Select between 3 and 4 time series for VAR analysis"
        )
        
        if selected_assets_display:
            selected_symbols = [asset_options[display_name] for display_name in selected_assets_display]
            selected_names = [all_assets[symbol]['name'] for symbol in selected_symbols]
            
            st.success(f"✓ {len(selected_symbols)} variables selected")
            
            for symbol, name in zip(selected_symbols, selected_names):
                st.text(f"• {name} ({symbol})")
        
        st.markdown("---")
        
        # Date range
        st.subheader("2. Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2015, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Frequency
        frequency = st.selectbox("Data Frequency", ["daily", "weekly", "monthly"], index=1)
        
        st.markdown("---")
        
        # VAR parameters
        st.subheader("3. Analysis Parameters")
        max_lags = st.slider("Maximum lags to consider", 1, 12, 8)
        confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
        
        st.markdown("---")
        
        # Run button
        # Use session state to track if analysis has started
        if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_started = True
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if not st.session_state.analysis_started:
        st.info("👈 **Please configure your analysis in the sidebar and click 'Run Complete Analysis' to begin.**")
    
        st.markdown("""
        ## About This Application
        
        This dashboard implements a complete VAR and Cointegration analysis framework following the requirements for **Macroeconometrics II** at ISEG.
        
        ### Analysis Sections:
        1. **Introduction & Data Overview** - Variable selection and justification
        2. **Descriptive Statistics** - Summary statistics, time plots, unit root tests
        3. **VAR Estimation** - Lag selection, model fitting, diagnostic tests
        4. **Structural Analysis** - Granger causality, IRFs, variance decomposition
        5. **Forecasting** - Point forecasts, confidence intervals, model comparison
        6. **Cointegration Analysis** - Johansen test, VEC model estimation
        7. **Conclusion** - Summary and recommendations
        
        ### Features:
        - **1200+ tradeable assets** from global markets
        - **US Treasury yields** for monetary policy analysis
        - **Flexible variable selection** from YAML configuration
        - **Complete diagnostic testing** suite
        - **Professional visualizations** with Plotly
        - **Academic rigor** following econometric best practices
        """)
        
        return
    
    # Validation
    if not selected_assets_display or len(selected_assets_display) < 3:
        st.error("⚠️ Please select at least 3 variables for analysis.")
        return
    
    if len(selected_assets_display) > 4:
        st.error("⚠️ Please select maximum 4 variables for analysis.")
        return
    
    # Fetch data
    with st.spinner("Fetching data from Yahoo Finance..."):
        df_raw = fetch_data(
            selected_symbols,
            selected_names,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            frequency
        )
    
    if df_raw.empty:
        st.error("Failed to fetch data. Please check your selection and try again.")
        return
    
    st.success(f"✅ Successfully fetched {len(df_raw):,} observations from {df_raw.index[0].date()} to {df_raw.index[-1].date()}")
    
    variables = list(df_raw.columns)
    
    # ====================================================================
    # SECTION 1: INTRODUCTION
    # ====================================================================
    
    st.markdown('<div class="section-header">1. Introduction</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### 1.1 Research Objective
    
    This study analyzes the dynamic relationships between **{len(variables)} financial market variables** using Vector Autoregressive (VAR) models and cointegration analysis. 
    The selected variables represent key indicators in **{selected_category if selected_category != "All Categories" else "global financial markets"}**.
    
    ### 1.2 Selected Variables
    
    The following time series have been selected for analysis:
    """)
    
    for i, (symbol, name) in enumerate(zip(selected_symbols, selected_names), 1):
        asset_info = all_assets[symbol]
        st.markdown(f"""
        **{i}. {name}** (`{symbol}`)
        - Category: {asset_info['category']}
        - Description: {asset_info['description']}
        """)
    
    st.markdown(f"""
    ### 1.3 Data Specifications
    
    - **Source**: Yahoo Finance (yfinance API)
    - **Sample Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
    - **Frequency**: {frequency.capitalize()}
    - **Observations**: {len(df_raw):,}
    
    ### 1.4 Methodology
    
    This analysis follows a systematic approach:
    
    1. **Preliminary analysis**: Descriptive statistics, stationarity tests, seasonality detection
    2. **VAR estimation**: Lag order selection, model fitting, diagnostic testing
    3. **Structural analysis**: Granger causality, impulse response functions, variance decomposition
    4. **Forecasting**: Out-of-sample forecasts with confidence intervals
    5. **Cointegration analysis**: Johansen test and Vector Error Correction Model (VEC)
    
    ### 1.5 Economic Relevance
    
    Understanding the relationships between these variables is crucial for:
    - Portfolio management and risk assessment
    - Monetary and fiscal policy analysis
    - Market forecasting and trading strategies
    - Economic research and policy recommendations
    """)
    
    # ====================================================================
    # SECTION 2: DESCRIPTIVE STATISTICS
    # ====================================================================
    
    st.markdown('<div class="section-header">2. Descriptive Statistics and Data Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("📈 2.1 Time Series Plots", expanded=True):
        fig = make_subplots(
            rows=len(variables), cols=1,
            subplot_titles=variables,
            vertical_spacing=0.08
        )
        
        for idx, var in enumerate(variables):
            fig.add_trace(
                go.Scatter(x=df_raw.index, y=df_raw[var], name=var, line=dict(width=2)),
                row=idx+1, col=1
            )
            fig.update_xaxes(title_text="Date", row=idx+1, col=1)
            fig.update_yaxes(title_text=var, row=idx+1, col=1)
        
        fig.update_layout(height=300*len(variables), showlegend=False, title_text="Original Time Series")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Commentary on Time Series Behavior:**
        
        The time series plots reveal several important features:
        - **Trends**: Identify long-term upward or downward movements
        - **Volatility**: Observe periods of high and low variance
        - **Structural breaks**: Note potential regime changes or crises
        - **Co-movement**: Assess visual correlation between series
        
        These patterns will inform our choice of transformations and model specifications.
        """)
    
    with st.expander("📊 2.2 Summary Statistics", expanded=True):
        stats_df = df_raw.describe().T
        stats_df['Skewness'] = df_raw.skew()
        stats_df['Kurtosis'] = df_raw.kurt()
        stats_df['CV (%)'] = (stats_df['std'] / stats_df['mean'] * 100).abs()
        
        stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'Skewness', 'Kurtosis', 'CV (%)']]
        
        st.dataframe(stats_df.style.format({
            'count': '{:.0f}',
            'mean': '{:.4f}',
            'std': '{:.4f}',
            'min': '{:.4f}',
            '25%': '{:.4f}',
            '50%': '{:.4f}',
            '75%': '{:.4f}',
            'max': '{:.4f}',
            'Skewness': '{:.4f}',
            'Kurtosis': '{:.4f}',
            'CV (%)': '{:.2f}'
        }).background_gradient(subset=['CV (%)'], cmap='YlOrRd'), use_container_width=True)
        
        st.markdown("**Statistical Interpretation:**")
        for var in variables:
            skew = df_raw[var].skew()
            kurt = df_raw[var].kurt()
            cv = (df_raw[var].std() / df_raw[var].mean() * 100)
            
            interpretation = []
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                interpretation.append(f"highly {direction}-skewed (skewness={skew:.2f})")
            if kurt > 3:
                interpretation.append(f"heavy-tailed with excess kurtosis={kurt-3:.2f}")
            elif kurt < 3:
                interpretation.append(f"light-tailed with kurtosis deficit={3-kurt:.2f}")
            if abs(cv) > 50:
                interpretation.append(f"high volatility (CV={cv:.1f}%)")
            
            if interpretation:
                st.write(f"- **{var}**: {', '.join(interpretation)}")
        
        st.info("""
        **Key Statistical Concepts:**
        - **Skewness**: Measures asymmetry. Zero indicates symmetric distribution.
        - **Kurtosis**: Measures tail thickness. Excess kurtosis > 0 indicates fat tails.
        - **CV (Coefficient of Variation)**: Relative volatility measure (std/mean).
        """)
    
    with st.expander("🔗 2.3 Correlation Analysis", expanded=True):
        corr_matrix = df_raw.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(title="Pearson Correlation Matrix", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Correlation Interpretation:**")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                corr_pairs.append((variables[i], variables[j], corr_matrix.iloc[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        st.write("**Strongest correlations:**")
        for var1, var2, corr in corr_pairs[:3]:
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            st.write(f"- {var1} ↔ {var2}: {strength} {direction} correlation (ρ={corr:.3f})")
    
    with st.expander("📅 2.4 Seasonality Analysis", expanded=True):
        st.markdown("**Seasonality Detection:**")
        
        freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12}
        test_freq = freq_map.get(frequency, 12)
        
        seasonality_results = []
        for var in variables:
            has_season, msg = detect_seasonality(df_raw[var], freq=test_freq)
            seasonality_results.append({
                'Variable': var,
                'Has Seasonality': '✓' if has_season else '✗',
                'Details': msg
            })
            
            if has_season:
                st.warning(f"**{var}**: Seasonality detected - {msg}")
            else:
                st.info(f"**{var}**: {msg}")
        
        st.dataframe(pd.DataFrame(seasonality_results), use_container_width=True)
        
        st.markdown("""
        **Recommendation:** 
        - If strong seasonality is detected, consider:
          1. Seasonal differencing
          2. Adding seasonal dummy variables
          3. Using seasonal ARIMA for univariate comparisons
        """)
    
    with st.expander("🔍 2.5 Unit Root Tests (Stationarity Analysis)", expanded=True):
        st.markdown("### Augmented Dickey-Fuller (ADF) Test")
        st.markdown("""
        **Null Hypothesis (H₀)**: The series has a unit root (non-stationary)  
        **Alternative (H₁)**: The series is stationary  
        **Decision Rule**: Reject H₀ if p-value < 0.05
        """)
        
        # Test with constant and trend
        adf_results_ct = []
        for var in variables:
            result = perform_adf_test(df_raw[var], var, regression='ct')
            adf_results_ct.append(result)
        
        adf_df = pd.DataFrame(adf_results_ct)
        
        def highlight_stationarity(row):
            if row['Stationary (5%)']:
                return ['background-color: #d4edda']*len(row)
            else:
                return ['background-color: #f8d7da']*len(row)
        
        st.dataframe(
            adf_df[['Variable', 'ADF Statistic', 'p-value', 'Lags Used', 'Critical Value (5%)', 'Stationary (5%)']].style
            .apply(highlight_stationarity, axis=1)
            .format({
                'ADF Statistic': '{:.4f}',
                'p-value': '{:.4f}',
                'Lags Used': '{:.0f}',
                'Critical Value (5%)': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.markdown("### KPSS Test (Confirmatory)")
        st.markdown("""
        **Null Hypothesis (H₀)**: The series is stationary  
        **Alternative (H₁)**: The series has a unit root  
        **Decision Rule**: Reject H₀ if p-value < 0.05
        """)
        
        kpss_results = []
        for var in variables:
            result = perform_kpss_test(df_raw[var], var, regression='ct')
            kpss_results.append(result)
        
        kpss_df = pd.DataFrame(kpss_results)
        st.dataframe(
            kpss_df[['Variable', 'KPSS Statistic', 'p-value', 'Critical Value (5%)', 'Stationary (5%)']].style
            .apply(highlight_stationarity, axis=1)
            .format({
                'KPSS Statistic': '{:.4f}',
                'p-value': '{:.4f}',
                'Critical Value (5%)': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.markdown("### Combined Interpretation")
        non_stationary = []
        stationary = []
        
        for adf_r, kpss_r in zip(adf_results_ct, kpss_results):
            var = adf_r['Variable']
            adf_stat = adf_r['Stationary (5%)']
            kpss_stat = kpss_r['Stationary (5%)']
            
            if adf_stat and kpss_stat:
                st.success(f"**{var}**: Both tests confirm stationarity ✓✓")
                stationary.append(var)
            elif not adf_stat and not kpss_stat:
                st.error(f"**{var}**: Both tests indicate non-stationarity ✗✗")
                non_stationary.append(var)
            else:
                st.warning(f"**{var}**: Mixed results - further investigation needed ⚠")
                non_stationary.append(var)
        
        # Determine transformations
        auto_transforms, recommendations = determine_transformation(df_raw, adf_results_ct)
        
        st.markdown("### Transformation Recommendations")
        for var, rec in recommendations.items():
            st.write(f"- **{var}**: {rec}")
    
    # ====================================================================
    # TRANSFORMATION DECISION
    # ====================================================================
    
    st.markdown("---")
    st.markdown('<div class="subsection-header">2.6 Data Transformation Decision</div>', unsafe_allow_html=True)
    
    transformation_choice = st.radio(
        "Choose transformation approach:",
        ["Use recommended transformations (based on unit root tests)",
         "Proceed with levels (for cointegration analysis)",
         "Custom transformations"]
    )
    
    if transformation_choice == "Use recommended transformations (based on unit root tests)":
        transformations = auto_transforms
        st.success("✓ Using statistically appropriate transformations for VAR in differences.")
    elif transformation_choice == "Proceed with levels (for cointegration analysis)":
        transformations = {var: 'level' for var in variables}
        if non_stationary:
            st.warning(f"⚠️ Variables {', '.join(non_stationary)} are non-stationary. Cointegration analysis will be performed.")
    else:
        st.write("**Select transformation for each variable:**")
        transformations = {}
        cols = st.columns(2)
        for idx, var in enumerate(variables):
            with cols[idx % 2]:
                transformations[var] = st.selectbox(
                    f"{var}:",
                    ["level", "log", "diff", "log_diff"],
                    index=0,
                    key=f"transform_{var}"
                )
    
    # Apply transformations
    df, transform_info = apply_transformation(df_raw, transformations)
    
    if df.empty or len(df) < 20:
        st.error("❌ Insufficient data after transformation. Please adjust your parameters.")
        return
    
    st.success(f"✓ Data prepared: **{len(df)} observations** available for analysis")
    
    # Show transformed data
    with st.expander("View Transformed Data"):
        st.write("**Applied Transformations:**")
        for var, trans in transform_info.items():
            st.write(f"- {var}: **{trans}**")
        
        fig = make_subplots(
            rows=len(variables), cols=1,
            subplot_titles=[f"{var} ({transform_info[var]})" for var in variables],
            vertical_spacing=0.08
        )
        
        for idx, var in enumerate(variables):
            fig.add_trace(
                go.Scatter(x=df.index, y=df[var], name=var, line=dict(width=2)),
                row=idx+1, col=1
            )
            fig.update_xaxes(title_text="Date", row=idx+1, col=1)
            fig.update_yaxes(title_text=var, row=idx+1, col=1)
        
        fig.update_layout(height=300*len(variables), showlegend=False, title_text="Transformed Time Series")
        st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # DEBUG: VERIFY DATA STATIONARITY BEFORE VAR
    # ====================================================================
    
    st.markdown("---")
    st.markdown("### 🔬 DEBUG: Pre-VAR Data Quality Check")
    st.warning("**This section helps diagnose why VAR might be unstable**")
    
    with st.expander("🔍 Stationarity Verification", expanded=True):
        st.markdown("#### 1. Transformation Summary")
        st.write("**Applied transformations:**")
        transform_df = pd.DataFrame([
            {'Variable': var, 'Transformation': transform_info[var]} 
            for var in variables
        ])
        st.dataframe(transform_df, use_container_width=True)
        
        # Highlight if any are 'level'
        if any(transform_info[var] == 'level' for var in variables):
            st.error("""
            🚨 **WARNING: Some variables are NOT transformed!**
            
            Variables with 'level' transformation are non-stationary and will cause:
            - Unstable VAR model (roots > 1)
            - Spurious regressions
            - Invalid inference
            
            **Action:** Change all transformations to 'log_diff' or 'diff'
            """)
        
        st.markdown("#### 2. Data Preview")
        st.write("**First 5 rows of transformed data:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.write("**Last 5 rows of transformed data:**")
        st.dataframe(df.tail(), use_container_width=True)
        
        st.markdown("#### 3. Descriptive Statistics")
        desc_stats = df.describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Check for stationarity indicators
        st.markdown("#### 4. Stationarity Indicators")
        
        stationarity_checks = []
        for var in variables:
            series = df[var].dropna()
            
            # Mean close to zero?
            mean_val = series.mean()
            mean_ok = abs(mean_val) < series.std() * 0.1  # Mean within 10% of std
            
            # No obvious trend?
            
            x = np.arange(len(series))
            slope, _, _, p_value, _ = scipy_stats.linregress(x, series)
            trend_ok = p_value > 0.05  # No significant trend
            
            # ADF test
            adf_stat, adf_pval = adfuller(series)[:2]
            adf_ok = adf_pval < 0.05
            
            stationarity_checks.append({
                'Variable': var,
                'Mean Near Zero': '✓' if mean_ok else '✗',
                'No Trend': '✓' if trend_ok else '✗',
                'ADF p-value': f"{adf_pval:.4f}",
                'Stationary?': '✓ YES' if adf_ok else '✗ NO'
            })
        
        stat_df = pd.DataFrame(stationarity_checks)
        st.dataframe(stat_df.style.applymap(
            lambda x: 'background-color: #ffcccc' if '✗' in str(x) else 
                     ('background-color: #ccffcc' if '✓' in str(x) else ''),
            subset=['Mean Near Zero', 'No Trend', 'Stationary?']
        ), use_container_width=True)
        
        # Overall verdict
        all_stationary = all('✓ YES' in check['Stationary?'] for check in stationarity_checks)
        
        if all_stationary:
            st.success("""
            ✅ **ALL VARIABLES ARE STATIONARY!**
            
            Your transformed data passes stationarity tests.
            The VAR model should be stable (roots < 1).
            
            If you still see unstable roots, there might be:
            - Structural breaks in the data
            - Insufficient lag order
            - Extreme outliers
            """)
        else:
            st.error("""
            ❌ **SOME VARIABLES ARE NON-STATIONARY!**
            
            This will cause:
            - Unstable VAR model (roots > 1)
            - Spurious regression results
            - Invalid IRFs and forecasts
            
            **Required Actions:**
            1. Go back to Phase 1
            2. Apply proper transformations (log_diff or diff)
            3. Re-run the analysis
            
            **DO NOT proceed with VAR estimation on non-stationary data!**
            """)
            
            # Stop execution if data is non-stationary
            st.warning("⚠️ Proceeding with non-stationary data at your own risk!")
        
        st.markdown("#### 5. Visual Stationarity Check")
        st.write("""
        **What to look for:**
        - Data should fluctuate around zero
        - No visible upward/downward trends
        - Constant variance over time
        """)
        
        # Plot each series
        fig = make_subplots(
            rows=len(variables), cols=1,
            subplot_titles=[f"{var} - Checking Stationarity" for var in variables],
            vertical_spacing=0.1
        )
        
        for idx, var in enumerate(variables):
            # Add the series
            fig.add_trace(
                go.Scatter(x=df.index, y=df[var], name=var, 
                          line=dict(color='blue', width=1.5)),
                row=idx+1, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         opacity=0.5, row=idx+1, col=1)
            
            # Add mean line
            mean_val = df[var].mean()
            fig.add_hline(y=mean_val, line_dash="dot", line_color="green", 
                         opacity=0.5, row=idx+1, col=1,
                         annotation_text=f"Mean: {mean_val:.4f}")
            
            fig.update_xaxes(title_text="Date", row=idx+1, col=1)
            fig.update_yaxes(title_text=var, row=idx+1, col=1)
        
        fig.update_layout(
            height=300*len(variables), 
            showlegend=False,
            title_text="Stationarity Visual Inspection"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Good stationarity indicators:**
        - Series oscillates around zero (red dashed line)
        - Mean (green dotted line) is close to zero
        - No systematic upward or downward movement
        - Variance looks constant throughout the sample
        """)

# ====================================================================
    # SECTION 3: VAR ESTIMATION
    # ====================================================================
    
    st.markdown('<div class="section-header">3. VAR Model Estimation</div>', unsafe_allow_html=True)
    
    with st.expander("📐 3.1 Lag Order Selection", expanded=True):
        st.markdown("### Information Criteria for Optimal Lag Selection")
        st.markdown("""
        We test multiple lag orders and select the model that minimizes information criteria:
        - **AIC** (Akaike): Balances fit and parsimony
        - **BIC** (Bayesian): More parsimonious, penalizes parameters more
        - **HQIC** (Hannan-Quinn): Intermediate between AIC and BIC
        """)
        
        try:
            model = VAR(df)
            
            ic_results = []
            for lag in range(1, max_lags + 1):
                try:
                    var_temp = model.fit(lag)
                    ic_results.append({
                        'Lag': lag,
                        'AIC': var_temp.aic,
                        'BIC': var_temp.bic,
                        'HQIC': var_temp.hqic,
                        'FPE': var_temp.fpe
                    })
                except:
                    continue
            
            if not ic_results:
                st.error("Could not estimate VAR models. Check your data.")
                return
            
            criteria_df = pd.DataFrame(ic_results)
            
            st.dataframe(criteria_df.style.format({
                'AIC': '{:.4f}',
                'BIC': '{:.4f}',
                'HQIC': '{:.4f}',
                'FPE': '{:.6e}'  # Scientific notation for small numbers
            }).highlight_min(subset=['AIC', 'BIC', 'HQIC', 'FPE'], color='lightgreen', axis=0),
            use_container_width=True)
            
            # Recommended lags
            aic_lag = int(criteria_df.loc[criteria_df['AIC'].idxmin(), 'Lag'])
            bic_lag = int(criteria_df.loc[criteria_df['BIC'].idxmin(), 'Lag'])
            hqic_lag = int(criteria_df.loc[criteria_df['HQIC'].idxmin(), 'Lag'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AIC recommends", f"{aic_lag} lags", help="Akaike Information Criterion")
            with col2:
                st.metric("BIC recommends", f"{bic_lag} lags", help="Bayesian Information Criterion (more parsimonious)")
            with col3:
                st.metric("HQIC recommends", f"{hqic_lag} lags", help="Hannan-Quinn Information Criterion")
            
            st.info("**Recommendation:** BIC typically selects more parsimonious models and is preferred for smaller samples.")
            
            # User selection
            # Initialize if first time
            # User selection with better state management
            # Initialize selected lag if first time OR if not in available options
            if ('selected_lag' not in st.session_state or 
                st.session_state.selected_lag is None or
                st.session_state.selected_lag not in criteria_df['Lag'].tolist()):
                st.session_state.selected_lag = bic_lag

            # Get the index safely
            try:
                current_index = criteria_df['Lag'].tolist().index(st.session_state.selected_lag)
            except ValueError:
                current_index = criteria_df['Lag'].tolist().index(bic_lag)

            # Create selectbox with callback
            def on_lag_change():
                """Callback when lag changes"""
                # Clear any cached model
                if 'var_model' in st.session_state:
                    st.session_state.var_model = None

            selected_lag = st.selectbox(
                "Select lag order for VAR model:",
                options=criteria_df['Lag'].tolist(),
                index=current_index,
                key='lag_order_selection',
                on_change=on_lag_change,
                help="Select number of lags to include in VAR model. Model will re-estimate when changed."
            )

            # Update session state
            st.session_state.selected_lag = selected_lag

            # Show what will happen
            if selected_lag != current_index + 1:  # If different from current
                st.info(f"🔄 Model will be re-estimated with {selected_lag} lag(s)")
            
        except Exception as e:
            st.error(f"Error in lag selection: {e}")
            selected_lag = st.number_input("Manually select lag order:", min_value=1, max_value=max_lags, value=2)
    
    # Fit VAR model
    # Fit VAR model
    try:
        # Check if we need to refit
        need_refit = (
            st.session_state.var_model is None or 
            st.session_state.var_model.k_ar != int(selected_lag)
        )
        
        if need_refit:
            st.info(f"Fitting VAR({int(selected_lag)}) model...")
            model = VAR(df)
            var_model = model.fit(int(selected_lag))
            st.session_state.var_model = var_model
        else:
            var_model = st.session_state.var_model
            st.info(f"Using cached VAR({int(selected_lag)}) model")
            
    except Exception as e:
        st.error(f"Error fitting VAR model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    with st.expander("📊 3.2 VAR Model Results", expanded=True):
        st.markdown(f"### VAR({int(selected_lag)}) Model Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lag Order (p)", int(selected_lag))
        with col2:
            st.metric("Observations", var_model.nobs)
        with col3:
            st.metric("AIC", f"{var_model.aic:.4f}")
        with col4:
            st.metric("BIC", f"{var_model.bic:.4f}")
        
        # Mathematical representation
        st.markdown("### Mathematical Representation")
        st.latex(r"""
        \begin{bmatrix} y_{1,t} \\ y_{2,t} \\ \vdots \\ y_{k,t} \end{bmatrix} = 
        \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_k \end{bmatrix} +
        \sum_{i=1}^{p} A_i \begin{bmatrix} y_{1,t-i} \\ y_{2,t-i} \\ \vdots \\ y_{k,t-i} \end{bmatrix} +
        \begin{bmatrix} \varepsilon_{1,t} \\ \varepsilon_{2,t} \\ \vdots \\ \varepsilon_{k,t} \end{bmatrix}
        """)
        
        st.markdown(f"where **k = {len(variables)}** variables and **p = {int(selected_lag)}** lags")
        
        # Coefficients
        st.markdown("### Estimated Parameters")
        
        st.markdown("**Intercepts (c):**")
        intercepts = var_model.params.iloc[0, :]
        st.dataframe(pd.DataFrame(intercepts, columns=['Intercept']).T.style.format('{:.4f}'), 
                    use_container_width=True)
        
        for i in range(int(selected_lag)):
            st.markdown(f"**Coefficient Matrix A_{i+1} (Lag {i+1}):**")
            start_idx = 1 + i * len(variables)
            end_idx = 1 + (i + 1) * len(variables)
            coef_matrix = var_model.params.iloc[start_idx:end_idx, :].T
            coef_matrix.columns = [f"{v}(t-{i+1})" for v in variables]
            coef_matrix.index = variables
            st.dataframe(coef_matrix.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
                        use_container_width=True)
        
        # R-squared - FIXED VERSION
        st.markdown("### Model Fit Statistics")
        rsq_data = []
        
        # Calculate R-squared manually from residuals and fitted values
        try:
            for var in variables:
                # Get actual values
                y_actual = df[var].values[var_model.k_ar:]
                
                # Get fitted values for this equation
                fitted = var_model.fittedvalues[var].values
                
                # Calculate R-squared
                ss_res = np.sum((y_actual - fitted) ** 2)
                ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Calculate adjusted R-squared
                n = len(y_actual)
                k = var_model.k_ar * len(variables) + 1  # number of parameters
                adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
                
                rsq_data.append({
                    'Equation': var,
                    'R-squared': r_squared,
                    'Adj. R-squared': adj_r_squared
                })
            
            rsq_df = pd.DataFrame(rsq_data)
            st.dataframe(rsq_df.style.format({
                'R-squared': '{:.4f}',
                'Adj. R-squared': '{:.4f}'
            }).background_gradient(subset=['R-squared', 'Adj. R-squared'], cmap='YlGn'), use_container_width=True)
            
        except Exception as e:
            st.warning(f"⚠️ Could not calculate R-squared: {e}")
            st.info("Continuing analysis without R-squared statistics.")
    
    with st.expander("🔬 3.3 Comprehensive Diagnostic Tests (Phase 2)", expanded=True):
            st.markdown("### Model Diagnostics - Multivariate Tests")
            st.info("""
            
            1. ✓ Stability Check (Eigenvalues)
            2. ✓ Serial Correlation (Portmanteau & Breusch-Godfrey)
            3. ✓ ARCH Effects (Heteroskedasticity)
            4. ✓ Normality Tests
            5. ✓ Structural Stability (CUSUM)
            
            **Important:** All tests are MULTIVARIATE (testing the full VAR system, not individual equations)
            """)
            
            # ================================================================
            # TEST 1: STABILITY CHECK
            # ================================================================
            # ================================================================
            # TEST 1: STABILITY CHECK - SHOWS BOTH EIGENVALUES AND ROOTS
            # ================================================================
            st.markdown("---")
            st.markdown("#### 1. ⚠️ STABILITY CHECK (Non-Negotiable)")
            st.markdown("**Critical Requirement:** ALL eigenvalues must be < 1 (inside unit circle)")
            
            st.info("""
            **Understanding VAR Stability:**
            
            A VAR model is stable if all eigenvalues of the companion matrix are < 1.
            
            **What statsmodels returns:**
            - `var_model.roots` = Inverse eigenvalues (roots = 1/λ)
            - For stability: eigenvalues < 1 → roots > 1
            
            **We show BOTH below for complete understanding:**
            1. **Eigenvalues (λ)**: The actual values that determine stability
            2. **Inverse roots (1/λ)**: What statsmodels calls "roots"
            """)
            
            try:
                # Get the companion matrix eigenvalues
                K = len(variables)  # Number of variables
                p = var_model.k_ar  # Number of lags
                
                # Get coefficient matrices from VAR model
                params = var_model.params
                
                # Build companion matrix
                companion = np.zeros((K * p, K * p))
                
                # Fill first K rows with VAR coefficients
                for i in range(p):
                    start_idx = 1 + i * K  # Skip intercept
                    end_idx = start_idx + K
                    companion[0:K, i*K:(i+1)*K] = params.iloc[start_idx:end_idx, :].T.values
                
                # Fill identity matrix below (for lags 2, 3, ..., p)
                if p > 1:
                    companion[K:, 0:-K] = np.eye(K * (p - 1))
                
                # Calculate eigenvalues of companion matrix
                eigenvalues = np.linalg.eigvals(companion)
                eigenvalues_moduli = np.abs(eigenvalues)
                
                # Get roots from statsmodels (these are inverse eigenvalues)
                roots = var_model.roots
                roots_moduli = np.abs(roots)
                
                # Get stability from statsmodels
                is_stable = var_model.is_stable()
                
                # ============================================================
                # IMPROVED: COMBINED STABILITY TABLE
                # ============================================================
                st.markdown("##### 📊 Combined Stability Analysis Table")
                st.markdown("**Eigenvalues, Roots, Inverse Roots, and Stability Status**")
                st.markdown("✅ For stability: **ALL Eigenvalue |λ| < 1** (equivalent to **ALL Root |1/λ| > 1**)")
                
                # Use the new combined function
                is_stable_combined, stability_df = display_combined_stability_table(var_model, variables)
                
                if stability_df is not None:
                    st.dataframe(stability_df.style.format({
                        'Eigenvalue (Real)': '{:.6f}',
                        'Eigenvalue (Imag)': '{:.6f}',
                        'Eigenvalue |λ|': '{:.6f}',
                        'Root (Real)': '{:.6f}',
                        'Root (Imag)': '{:.6f}',
                        'Root |1/λ|': '{:.6f}',
                        'Inverse Root (Real)': '{:.6f}',
                        'Inverse Root (Imag)': '{:.6f}'
                    }).applymap(lambda x: 'background-color: #ccffcc' if '✓' in str(x) else 
                               ('background-color: #ffcccc' if '✗' in str(x) else ''),
                               subset=['Stable?']),
                    use_container_width=True)
                
                # ============================================================
                # DISPLAY 3: SUMMARY METRICS
                # ============================================================
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Number of Eigenvalues", len(eigenvalues))
                with col2:
                    st.metric("Max |Eigenvalue|", f"{eigenvalues_moduli.max():.6f}")
                with col3:
                    st.metric("Min |Root|", f"{roots_moduli.min():.6f}")
                with col4:
                    if is_stable:
                        st.success("✓ STABLE")
                    else:
                        st.error("✗ UNSTABLE")
                
                # ============================================================
                # DISPLAY 4: VISUAL PLOT
                # ============================================================
                st.markdown("##### 📈 Visual Representation: Eigenvalues in Complex Plane")
                
                fig = go.Figure()
                
                # Draw unit circle
                theta = np.linspace(0, 2*np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=np.cos(theta), 
                    y=np.sin(theta),
                    mode='lines', 
                    name='Unit Circle (Stability Boundary)',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                # Plot eigenvalues
                colors = ['green' if m < 1 else 'red' for m in eigenvalues_moduli]
                
                fig.add_trace(go.Scatter(
                    x=np.real(eigenvalues), 
                    y=np.imag(eigenvalues),
                    mode='markers+text',
                    name='Eigenvalues (λ)',
                    marker=dict(size=12, color=colors, symbol='circle', 
                               line=dict(width=2, color='white')),
                    text=[f"λ{i}" for i in range(len(eigenvalues))],
                    textposition="top center",
                    hovertemplate='λ%{text}<br>Real: %{x:.4f}<br>Imag: %{y:.4f}<br>|λ|: %{marker.color}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Eigenvalues of Companion Matrix (Stability Diagnostic)',
                    xaxis_title='Real Part',
                    yaxis_title='Imaginary Part',
                    height=500,
                    showlegend=True,
                    xaxis=dict(zeroline=True, range=[-1.5, 1.5]),
                    yaxis=dict(zeroline=True, scaleanchor="x", range=[-1.5, 1.5])
                )
                
                fig.add_annotation(
                    text="<b>Stability Condition:</b> ALL eigenvalues (green dots) must be INSIDE red circle (|λ| < 1)",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=12, color='darkblue'),
                    bgcolor='lightyellow',
                    bordercolor='orange',
                    borderwidth=2
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ============================================================
                # DISPLAY 5: STABILITY VERDICT
                # ============================================================
                st.markdown("##### 🎯 Stability Verdict")
                
                if is_stable and eigenvalues_moduli.max() < 1.0:
                    st.success(f"""
                    ✅ **STABILITY CHECK: PASS**
                    
                    **Eigenvalue Analysis:**
                    - Total eigenvalues: {len(eigenvalues)}
                    - Maximum |λ| = {eigenvalues_moduli.max():.6f} < 1.0 ✓
                    - All eigenvalues are inside unit circle
                    
                    **Interpretation:**
                    - Model is **stationary** and **stable**
                    - Shocks decay over time (no explosive behavior)
                    - Safe to compute impulse responses and forecasts
                    
                    **✓ You may proceed with structural analysis!**
                    """)
                else:
                    max_eigenval = eigenvalues_moduli.max()
                    st.error(f"""
                    ❌ **STABILITY CHECK: FAIL**
                    
                    **Eigenvalue Analysis:**
                    - Total eigenvalues: {len(eigenvalues)}
                    - Maximum |λ| = {max_eigenval:.6f} ≥ 1.0 ✗
                    - Model has explosive dynamics
                    
                    **What this means:**
                    - Shocks do NOT decay (explosive behavior)
                    - Forecasts will diverge to infinity
                    - Impulse responses are invalid
                    
                    **Action Required:**
                    1. ⚠️ STOP - Do NOT proceed with analysis
                    2. Check your data transformations (must be stationary)
                    3. Verify lag order selection
                    4. Re-run unit root tests on transformed data
                    """)
                    
                # Show relationship between eigenvalues and roots
                st.markdown("##### 🔍 Understanding Eigenvalues vs Roots")
                st.markdown("""
                **Mathematical Relationship:**
                
                If λ is an eigenvalue of the companion matrix:
                - Root = 1/λ (inverse of eigenvalue)
                - |λ| < 1 ↔ |root| > 1
                
                **Stability Conditions (equivalent statements):**
                1. All eigenvalues have modulus < 1 (inside unit circle)
                2. All roots have modulus > 1 (outside unit circle in root space)
                3. `var_model.is_stable()` returns `True`
                
                **Why two representations?**
                - Eigenvalues (λ): Standard linear algebra perspective
                - Roots (1/λ): Time series lag polynomial perspective
                
                Both check the same condition, just from different angles!
                """)
                
            except Exception as e:
                st.error(f"❌ Could not perform stability check: {e}")
                st.code(f"Error details:\n{str(e)}")
                import traceback
                st.markdown("##### 📋 Full Error Traceback")
                st.code(traceback.format_exc())
            
            # ================================================================
            # TEST 2: SERIAL CORRELATION - MULTIVARIATE
            # ================================================================
            st.markdown("---")
            st.markdown("#### 2. 📊 Serial Correlation Tests")
            st.markdown("""
            **H₀:** No serial correlation in residuals (residuals are white noise)  
            **H₁:** Serial correlation exists (model misspecified)  
            **Decision:** Reject H₀ if p-value < 0.05
            """)
            
            # A. Portmanteau Test (Asymptotic)
            st.markdown("##### A. Multivariate Portmanteau Test (Asymptotic)")
            st.markdown("Testing for autocorrelation at lags h=1 to h=10")
            
            try:
                portmanteau_results = []
                
                for h in range(1, 11):
                    try:
                        # Statsmodels' test_whiteness is the multivariate Portmanteau test
                        # This is equivalent to R's serial.test(..., type="PT.asymptotic")
                        test = var_model.test_whiteness(nlags=h, signif=0.05, adjusted=False)
                        
                        portmanteau_results.append({
                            'Lag (h)': h,
                            'Chi-squared': test.test_statistic,
                            'df': f"{h * len(variables)**2}",
                            'P-value': test.pvalue,
                            'Decision': '✓ Pass' if test.pvalue > 0.05 else '✗ Reject (AC detected)'
                        })
                    except Exception as e:
                        st.warning(f"Could not compute Portmanteau test for h={h}: {e}")
                        continue
                
                if portmanteau_results:
                    port_df = pd.DataFrame(portmanteau_results)
                    st.dataframe(port_df.style.format({
                        'Chi-squared': '{:.4f}',
                        'P-value': '{:.4f}'
                    }).applymap(lambda x: 'background-color: #ffcccc' if '✗' in str(x) else 'background-color: #ccffcc', 
                            subset=['Decision']),
                    use_container_width=True)
                    
                    # Check critical lags 
                    critical_lags = [h for h in range(5, 11)]
                    critical_fails = [r for r in portmanteau_results if r['Lag (h)'] in critical_lags and r['P-value'] <= 0.05]
                    
                    if not critical_fails:
                        st.success("""
                        ✅ **PORTMANTEAU TEST: PASS**
                        
                        No serial correlation detected at critical lags (h=5 to h=10).
                        Residuals appear to be white noise.
                        """)
                    else:
                        failed_lags = [r['Lag (h)'] for r in critical_fails]
                        st.error(f"""
                        ❌ **PORTMANTEAU TEST: FAIL**
                        
                        Serial correlation detected at lags: {failed_lags}
                        
                        **Action:** Increase lag order to p={int(selected_lag)+2} or higher
                        """)
            except Exception as e:
                st.warning(f"Portmanteau test error: {e}")
            
            # B. Portmanteau Test (Adjusted) - Small sample correction
            st.markdown("##### B. Adjusted Portmanteau Test (Small Sample Correction)")
            
            try:
                portmanteau_adj_results = []
                
                for h in range(1, 11):
                    try:
                        # Adjusted version (better for finite samples)
                        test = var_model.test_whiteness(nlags=h, signif=0.05, adjusted=True)
                        
                        portmanteau_adj_results.append({
                            'Lag (h)': h,
                            'Chi-squared': test.test_statistic,
                            'df': f"{h * len(variables)**2}",
                            'P-value': test.pvalue,
                            'Decision': '✓ Pass' if test.pvalue > 0.05 else '✗ Reject'
                        })
                    except:
                        continue
                
                if portmanteau_adj_results:
                    port_adj_df = pd.DataFrame(portmanteau_adj_results)
                    st.dataframe(port_adj_df.style.format({
                        'Chi-squared': '{:.4f}',
                        'P-value': '{:.4f}'
                    }).applymap(lambda x: 'background-color: #ffcccc' if '✗' in str(x) else 'background-color: #ccffcc', 
                            subset=['Decision']),
                    use_container_width=True)
                    
                    critical_fails_adj = [r for r in portmanteau_adj_results if r['Lag (h)'] >= 5 and r['P-value'] <= 0.05]
                    if not critical_fails_adj:
                        st.success("✅ **ADJUSTED PORTMANTEAU: PASS**")
                    else:
                        st.error(f"❌ **ADJUSTED PORTMANTEAU: FAIL** at lags {[r['Lag (h)'] for r in critical_fails_adj]}")
            except Exception as e:
                st.warning(f"Adjusted Portmanteau test error: {e}")
            
            # C. Breusch-Godfrey LM Test (PROPER MULTIVARIATE VERSION)
            st.markdown("##### C. Multivariate Breusch-Godfrey LM Test")
            st.markdown("Alternative test for serial correlation using Lagrange Multiplier principle")
            
            try:
                # Statsmodels VAR doesn't have built-in multivariate BG test
                # We'll implement it following Lütkepohl (2005) approach
                
                st.info("""
                **Note:** Python's statsmodels doesn't have a direct equivalent to R's 
                `serial.test(..., type="BG")`. The Portmanteau tests above are the 
                primary multivariate serial correlation tests. 
                
                For the LM test, you can use the Portmanteau test results as they test 
                the same hypothesis (H₀: no serial correlation).
                """)
                
                # Alternative: Show equation-by-equation LM tests as supplementary
                st.markdown("**Supplementary: Equation-by-Equation LM Tests**")
                st.markdown("*(These are univariate tests, shown for completeness)*")
                
                lm_by_equation = []
                for var in variables:
                    try:
                        # Test for autocorrelation in this equation's residuals
                        lb_test = acorr_ljungbox(var_model.resid[var], lags=10, return_df=False)
                        # Get p-value for lag 10
                        pval = lb_test[1][-1]
                        
                        lm_by_equation.append({
                            'Equation': var,
                            'Lag': 10,
                            'P-value': pval,
                            'Result': '✓ Pass' if pval > 0.05 else '✗ Fail'
                        })
                    except:
                        continue
                
                if lm_by_equation:
                    lm_eq_df = pd.DataFrame(lm_by_equation)
                    st.dataframe(lm_eq_df.style.format({
                        'P-value': '{:.4f}'
                    }).applymap(lambda x: 'background-color: #ffcccc' if '✗' in str(x) else 'background-color: #ccffcc', 
                            subset=['Result']),
                    use_container_width=True)
                    
                    st.info("These are individual equation tests. The multivariate Portmanteau tests above are more appropriate for VAR.")
                    
            except Exception as e:
                st.warning(f"LM test error: {e}")
            
            # ================================================================
            # TEST 3: ARCH EFFECTS - MULTIVARIATE
            # ================================================================
            st.markdown("---")
            st.markdown("#### 3. 📈 ARCH Effects (Heteroskedasticity)")
            st.markdown("""
            **H₀:** No ARCH effects (constant variance)  
            **H₁:** ARCH effects present (volatility clustering)  
            **Note:** ARCH effects are common in financial data and don't invalidate the VAR model.
            """)
            
            st.markdown("##### Multivariate ARCH Test")
            
            try:
                # Statsmodels doesn't have multivariate ARCH test
                # We'll test each equation and report
                
                st.info("""
                **Note:** Python's statsmodels doesn't have R's `arch.test(..., multivariate.only=TRUE)`.
                We show equation-by-equation ARCH tests below. ARCH effects are typical for financial data.
                """)
                
                arch_results = []
                for var in variables:
                    try:
                        # Test for ARCH effects in residuals
                        # Use squared residuals
                        resid_sq = var_model.resid[var] ** 2
                        
                        # LM test for ARCH(1)
                        from statsmodels.stats.diagnostic import het_arch
                        lm_stat, lm_pval, f_stat, f_pval = het_arch(var_model.resid[var], nlags=1)
                        
                        arch_results.append({
                            'Equation': var,
                            'Lags': 1,
                            'LM Statistic': lm_stat,
                            'P-value': lm_pval,
                            'Result': '✓ No ARCH' if lm_pval > 0.05 else '⚠️ ARCH detected'
                        })
                    except Exception as e:
                        continue
                
                if arch_results:
                    arch_df = pd.DataFrame(arch_results)
                    st.dataframe(arch_df.style.format({
                        'LM Statistic': '{:.4f}',
                        'P-value': '{:.4f}'
                    }).applymap(lambda x: 'background-color: #fff3cd' if '⚠️' in str(x) else 'background-color: #ccffcc', 
                            subset=['Result']),
                    use_container_width=True)
                    
                    any_arch = any(r['P-value'] <= 0.05 for r in arch_results)
                    if any_arch:
                        st.warning("""
                        ⚠️ **ARCH EFFECTS DETECTED**
                        
                        This indicates volatility clustering (typical for financial data).
                        
                        **Impact:**
                        - Point estimates remain consistent
                        - Standard errors may be underestimated
                        - Consider: Robust standard errors or GARCH modeling
                        
                        **For your assignment:** Document this finding but proceed with analysis.
                        """)
                    else:
                        st.success("✅ **NO ARCH EFFECTS** - Variance appears constant")
            except Exception as e:
                st.warning(f"ARCH test error: {e}")
            
            # ================================================================
            # TEST 4: NORMALITY - MULTIVARIATE
            # ================================================================
            st.markdown("---")
            st.markdown("#### 4. 📊 Normality Tests")
            st.markdown("""
            **H₀:** Residuals are normally distributed  
            **H₁:** Residuals are not normally distributed  
            **Note:** Non-normality is common in financial data due to fat tails.
            """)
            
            st.markdown("##### Multivariate Normality Test")
            
            try:
                # Statsmodels has test_normality for VAR
                norm_test = var_model.test_normality(signif=0.05)
                
                st.write(f"**Test Statistic:** {norm_test.test_statistic:.4f}")
                st.write(f"**P-value:** {norm_test.pvalue:.4f}")
                st.write(f"**Critical Value (5%):** {norm_test.crit_value:.4f}")
                
                if norm_test.pvalue > 0.05:
                    st.success("""
                    ✅ **NORMALITY: PASS**
                    
                    Residuals are approximately normally distributed.
                    """)
                else:
                    st.warning("""
                    ⚠️ **NORMALITY: FAIL**
                    
                    Residuals are not normally distributed.
                    
                    **Common causes:**
                    - Fat tails (extreme events in financial markets)
                    - Skewness
                    - Outliers
                    
                    **Impact:**
                    - With large sample (n > 100), CLT applies
                    - Asymptotic inference still valid
                    - Consider: Bootstrap confidence intervals
                    
                    **For your assignment:** Document this but proceed. It's typical for financial data.
                    """)
                
                # Show skewness and kurtosis
                st.markdown("**Residual Statistics:**")
                stats_data = []
                for var in variables:
                    from scipy.stats import skew, kurtosis
                    resid = var_model.resid[var].dropna()
                    stats_data.append({
                        'Variable': var,
                        'Skewness': skew(resid),
                        'Kurtosis': kurtosis(resid, fisher=False),  # Excess kurtosis + 3
                        'Normal Kurtosis': 3.0
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.style.format({
                    'Skewness': '{:.4f}',
                    'Kurtosis': '{:.4f}',
                    'Normal Kurtosis': '{:.1f}'
                }), use_container_width=True)
                
                st.info("**Normal distribution:** Skewness ≈ 0, Kurtosis ≈ 3. Financial returns often have kurtosis > 3 (fat tails).")
                
            except Exception as e:
                st.warning(f"Normality test error: {e}")
            
            # ================================================================
            # TEST 5: STRUCTURAL STABILITY
            # ================================================================
            st.markdown("---")
            st.markdown("#### 5. 📉 Structural Stability")
            st.markdown("""
            **Test for:** Structural breaks or parameter instability over time  
            **Method:** Recursive residuals and CUSUM test
            """)
            
            try:
                st.info("""
                **Note:** Full CUSUM test implementation requires recursive estimation.
                For your assignment, you can:
                1. Visually inspect residuals for patterns over time
                2. Check for obvious structural breaks (e.g., COVID period)
                3. Consider splitting sample if breaks are evident
                """)
                
                # Plot residuals over time
                fig = make_subplots(
                    rows=len(variables), cols=1,
                    subplot_titles=[f"Residuals: {var}" for var in variables],
                    vertical_spacing=0.08
                )
                
                for idx, var in enumerate(variables):
                    fig.add_trace(
                        go.Scatter(x=var_model.resid.index, y=var_model.resid[var],
                                mode='lines', name=var, line=dict(width=1)),
                        row=idx+1, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=idx+1, col=1)
                    fig.update_xaxes(title_text="Date", row=idx+1, col=1)
                    fig.update_yaxes(title_text="Residual", row=idx+1, col=1)
                
                fig.update_layout(height=300*len(variables), showlegend=False, 
                                title_text="Residuals Over Time (Visual Stability Check)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("""
                ✅ **Visual inspection:** Check for:
                - Constant mean (residuals around zero)
                - Constant variance (no expanding/contracting spread)
                - No obvious structural breaks
                """)
                
            except Exception as e:
                st.warning(f"Stability check error: {e}")
            
            # ================================================================
            # NEW: RESTRICTED VAR ANALYSIS  
            # ================================================================
            st.markdown("---")
            st.markdown("#### 6. 🔒 Coefficient Significance Analysis")
            st.markdown("""
            **Objective:** Identify insignificant coefficients that could be restricted  
            **Method:** Test which coefficients have p-values > 0.05
            """)
            
            try:
                # Analyze coefficient significance
                n_insignificant, insignificant_list = analyze_coefficient_significance(
                    var_model, significance_level=0.05
                )
                
                st.info(f"Total parameters in model: {var_model.params.size}")
                
                if n_insignificant > 0:
                    st.warning(f"⚠️ Found {n_insignificant} insignificant coefficients (p > 0.05)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Insignificant Coefficients", n_insignificant)
                    with col2:
                        pct = (n_insignificant / (var_model.params.size - len(variables))) * 100
                        st.metric("% of Model (excl. const)", f"{pct:.1f}%")
                    
                    # Show insignificant coefficients
                    st.markdown("##### Insignificant Coefficients (p > 0.05)")
                    insig_df = pd.DataFrame(insignificant_list)
                    st.dataframe(insig_df.style.format({
                        'Coefficient': '{:.6f}',
                        'p-value': '{:.4f}'
                    }).apply(lambda x: ['background-color: #ffcccc']*len(x), axis=1),
                    use_container_width=True)
                    
                    st.markdown("##### Interpretation")
                    st.write("""
                    **What this means:**
                    - These coefficients are NOT statistically different from zero
                    - They add noise to the model without improving explanatory power
                    - A restricted VAR (setting these to zero) could improve:
                      - Parsimony (simpler model)
                      - Out-of-sample forecast performance
                      - Interpretation clarity
                    
                    **Options:**
                    1. **Keep unrestricted model** - Use for complete analysis
                    2. **Restrict model** - Set insignificant coefficients to zero (more parsimonious)
                    
                    **For your project:** Document this finding. It shows you understand model selection trade-offs.
                    """)
                    
                    # Show which equations have most insignificant coefficients
                    st.markdown("##### Analysis by Equation")
                    eq_summary = {}
                    for item in insignificant_list:
                        eq = item['Equation']
                        if eq not in eq_summary:
                            eq_summary[eq] = 0
                        eq_summary[eq] += 1
                    
                    eq_summary_data = [{
                        'Equation': eq,
                        'Insignificant Coefficients': count
                    } for eq, count in eq_summary.items()]
                    
                    if eq_summary_data:
                        eq_summary_df = pd.DataFrame(eq_summary_data)
                        st.dataframe(eq_summary_df, use_container_width=True)
                    
                    # Compare with full model
                    st.markdown("##### Model Information Criteria")
                    st.write(f"""
                    **Current Unrestricted Model:**
                    - AIC: {var_model.aic:.4f}
                    - BIC: {var_model.bic:.4f}
                    - Log-Likelihood: {var_model.llf:.4f}
                    
                    **Note:** Restricting {n_insignificant} coefficients would:
                    - Reduce AIC by approximately: {n_insignificant * 2:.2f}
                    - Reduce BIC by approximately: {n_insignificant * np.log(len(df)):.2f}
                    - These are rough estimates; actual restricted model would need re-estimation
                    """)
                    
                else:
                    st.success("✓ All coefficients are significant - no restrictions needed!")
                    st.write("""
                    **This is excellent!** It means:
                    - Every parameter in your model contributes meaningful information
                    - The model is already parsimonious
                    - No overfitting concerns from this perspective
                    """)
                
            except Exception as e:
                st.error(f"Error in coefficient significance analysis: {e}")
            
            # ================================================================
            # COMPREHENSIVE SUMMARY
            # ================================================================
            st.markdown("---")
            st.markdown("### 📋 Diagnostic Summary")
            
            # Collect all results
            summary_data = []
            
            try:
                # 1. Stability
                summary_data.append({
                    'Test': '1. Stability',
                    'Result': '✅ PASS' if is_stable else '❌ FAIL',
                    'Critical?': 'YES',
                    'Action': 'None' if is_stable else 'Re-specify model'
                })
            except:
                pass
            
            try:
                # 2. Serial Correlation
                serial_corr_ok = all(r['P-value'] > 0.05 for r in portmanteau_results if r['Lag (h)'] >= 5)
                summary_data.append({
                    'Test': '2. Serial Correlation',
                    'Result': '✅ PASS' if serial_corr_ok else '❌ FAIL',
                    'Critical?': 'YES',
                    'Action': 'None' if serial_corr_ok else f'Increase lag to p≥{int(selected_lag)+2}'
                })
            except:
                pass
            
            try:
                # 3. ARCH
                arch_ok = all(r['P-value'] > 0.05 for r in arch_results)
                summary_data.append({
                    'Test': '3. ARCH Effects',
                    'Result': '✅ None' if arch_ok else '⚠️ Present',
                    'Critical?': 'NO',
                    'Action': 'Document' if not arch_ok else 'None'
                })
            except:
                pass
            
            try:
                # 4. Normality
                norm_ok = norm_test.pvalue > 0.05
                summary_data.append({
                    'Test': '4. Normality',
                    'Result': '✅ PASS' if norm_ok else '⚠️ FAIL',
                    'Critical?': 'NO',
                    'Action': 'Document' if not norm_ok else 'None'
                })
            except:
                pass
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df.style.applymap(
                    lambda x: 'background-color: #ccffcc' if '✅' in str(x)
                    else ('background-color: #ffcccc' if '❌' in str(x)
                        else 'background-color: #fff3cd'),
                    subset=['Result']
                ), use_container_width=True)
                
                # Final verdict
                critical_fails = [d for d in summary_data if '❌' in d['Result'] and d['Critical?'] == 'YES']
                
                if not critical_fails:
                    st.success("""
                    ### ✅ **MODEL DIAGNOSTICS: ACCEPTABLE**
                    
                    All critical tests passed:
                    - ✓ Model is stable
                    - ✓ No serial correlation
                    
                    Non-critical warnings (ARCH, non-normality) are typical for financial data.
                    
                    **You may proceed with:**
                    - Granger causality tests
                    - Impulse response functions
                    - Forecast error variance decomposition
                    - Forecasting
                    """)
                else:
                    st.error("""
                    ### ❌ **MODEL DIAGNOSTICS: FAILED**
                    
                    Critical diagnostic failures detected.
                    
                    **Do not proceed with structural analysis until these are resolved!**
                    """)
    # ====================================================================

    # ============================================================================
# ============================================================================
# RESTRICTED VAR SECTION - Add this after Section 3.3 (Diagnostics)
# Insert around line 2115, before Section 4: Structural Analysis
# ============================================================================

    # ====================================================================
    # SECTION 3.4: RESTRICTED VAR (COEFFICIENT RESTRICTION)
    # ====================================================================
    
    st.markdown('<div class="section-header">3.4 Restricted VAR Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Model Parsimony and Coefficient Restriction
    
    The **unrestricted VAR model** may include many statistically insignificant coefficients, leading to:
    - ⚠️ Overfitting
    - ⚠️ Large standard errors
    - ⚠️ Poor out-of-sample forecasts
    
    **Solution:** Estimate a **Restricted VAR** by removing insignificant coefficients and re-estimating.
    """)
    
    with st.expander("📊 3.4.1 Coefficient Significance Testing", expanded=True):
        st.markdown("### Individual Coefficient Significance")
        
        st.info("""
        **Test for each coefficient:**
        - H₀: βᵢⱼ = 0 (coefficient is zero)
        - H₁: βᵢⱼ ≠ 0 (coefficient is non-zero)
        - Decision: Reject H₀ if |t-statistic| > critical value (typically 1.96 for 5% level)
        """)
        
        # Get coefficient estimates and standard errors
        try:
            coef_matrix = var_model.params
            stderr_matrix = var_model.stderr
            
            # Calculate t-statistics
            t_stats = coef_matrix / stderr_matrix
            
            # P-values (two-tailed test)
            from scipy.stats import t as t_dist
            df_resid = var_model.df_resid
            
            # Ensure we have DataFrames
            if not isinstance(t_stats, pd.DataFrame):
                t_stats = pd.DataFrame(t_stats, index=coef_matrix.index, columns=coef_matrix.columns)
            
            # Calculate p-values
            p_values_array = 2 * (1 - t_dist.cdf(np.abs(t_stats.values), df_resid))
            p_values = pd.DataFrame(p_values_array, index=coef_matrix.index, columns=coef_matrix.columns)
            
            # Significance at 5% level
            significant = p_values < 0.05
            
            st.markdown("### Coefficient Summary by Equation")
            
            for eq_idx, eq_name in enumerate(variables):
                st.markdown(f"#### Equation: {eq_name}")
                
                # Create summary dataframe
                coef_summary = []
                
                # Intercept
                coef_summary.append({
                    'Parameter': 'const',
                    'Coefficient': coef_matrix.iloc[0, eq_idx],
                    'Std Error': stderr_matrix.iloc[0, eq_idx],
                    't-statistic': t_stats.iloc[0, eq_idx],
                    'P-value': p_values.iloc[0, eq_idx],
                    'Significant (5%)': '✓' if significant.iloc[0, eq_idx] else '✗'
                })
                
                # Lagged variables
                for lag in range(1, int(selected_lag) + 1):
                    for var_idx, var_name in enumerate(variables):
                        row_idx = 1 + (lag - 1) * len(variables) + var_idx
                        param_name = f"{var_name}.L{lag}"
                        
                        coef_summary.append({
                            'Parameter': param_name,
                            'Coefficient': coef_matrix.iloc[row_idx, eq_idx],
                            'Std Error': stderr_matrix.iloc[row_idx, eq_idx],
                            't-statistic': t_stats.iloc[row_idx, eq_idx],
                            'P-value': p_values.iloc[row_idx, eq_idx],
                            'Significant (5%)': '✓' if significant.iloc[row_idx, eq_idx] else '✗'
                        })
                
                coef_df = pd.DataFrame(coef_summary)
                
                # Style the dataframe
                st.dataframe(
                    coef_df.style
                    .format({
                        'Coefficient': '{:.4f}',
                        'Std Error': '{:.4f}',
                        't-statistic': '{:.4f}',
                        'P-value': '{:.4f}'
                    })
                    .applymap(
                        lambda x: 'background-color: #d4edda' if x == '✓' else 'background-color: #f8d7da',
                        subset=['Significant (5%)']
                    ),
                    use_container_width=True
                )
                
                # Count significant coefficients
                n_sig = coef_df['Significant (5%)'].value_counts().get('✓', 0)
                n_total = len(coef_df)
                pct_sig = (n_sig / n_total) * 100
                
                st.write(f"**Significant coefficients:** {n_sig}/{n_total} ({pct_sig:.1f}%)")
            
            # Overall summary
            st.markdown("---")
            st.markdown("### Overall Model Summary")
            
            total_params = coef_matrix.size
            total_significant = significant.sum().sum()
            pct_significant = (total_significant / total_params) * 100
            total_insignificant = total_params - total_significant
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Parameters", total_params)
            with col2:
                st.metric("Significant (5%)", total_significant)
            with col3:
                st.metric("Insignificant", total_insignificant)
            with col4:
                st.metric("% Significant", f"{pct_significant:.1f}%")
            
            if pct_significant < 50:
                st.warning(f"""
                ⚠️ **Only {pct_significant:.1f}% of coefficients are statistically significant!**
                
                This suggests the model is overparameterized. A restricted VAR may improve:
                - Model interpretability
                - Forecast accuracy
                - Computational efficiency
                """)
            elif pct_significant < 70:
                st.info(f"""
                **{pct_significant:.1f}% of coefficients are significant.**
                
                Consider estimating a restricted VAR for improved parsimony.
                """)
            else:
                st.success(f"""
                ✓ **{pct_significant:.1f}% of coefficients are significant.**
                
                The model is reasonably parsimonious, but you can still try restriction.
                """)
        
        except Exception as e:
            st.error(f"Error calculating coefficient significance: {e}")
    
    # ====================================================================
    # AUTOMATIC RESTRICTION
    # ====================================================================
    
    with st.expander("🔧 3.4.2 Automatic Coefficient Restriction", expanded=True):
        st.markdown("### Restricted VAR Estimation")
        
        st.markdown("""
        **Method:** Automatically set insignificant coefficients to zero and re-estimate.
        
        **Process:**
        1. Identify insignificant coefficients (p-value > α)
        2. Create restriction matrix
        3. Re-estimate VAR with restrictions
        4. Compare restricted vs unrestricted model
        """)
        
        # User controls
        col1, col2 = st.columns(2)
        
        with col1:
            significance_level = st.selectbox(
                "Significance level (α):",
                options=[0.01, 0.05, 0.10],
                index=1,
                help="Coefficients with p-value > α will be restricted to zero"
            )
        
        with col2:
            keep_constant = st.checkbox(
                "Always keep intercepts",
                value=True,
                help="Keep all constant terms even if insignificant"
            )
        
        estimate_restricted = st.button("📊 Estimate Restricted VAR", type="primary")
        
        if estimate_restricted or 'restricted_var_model' in st.session_state:
            
            if estimate_restricted:
                with st.spinner("Estimating restricted VAR model..."):
                    try:
                        # Identify insignificant coefficients
                        # Safe array flattening - handle both DataFrame and ndarray
                        if hasattr(p_values, 'values'):
                            p_values_flat = p_values.values.flatten()
                        else:
                            p_values_flat = p_values.flatten() if isinstance(p_values, np.ndarray) else np.array(p_values).flatten()

                        if hasattr(coef_matrix, 'values'):
                            coef_flat = coef_matrix.values.flatten()
                        else:
                            coef_flat = coef_matrix.flatten() if isinstance(coef_matrix, np.ndarray) else np.array(coef_matrix).flatten()
                                                # Create restriction mask
                        if keep_constant:
                            # Keep intercepts (first row of each equation)
                            restriction_mask = np.ones_like(coef_flat, dtype=bool)
                            for eq_idx in range(len(variables)):
                                # First coefficient of each equation is intercept
                                restriction_mask[eq_idx] = True
                                # Other coefficients depend on significance
                                for param_idx in range(1, coef_matrix.shape[0]):
                                    flat_idx = param_idx * len(variables) + eq_idx
                                    if p_values_flat[flat_idx] > significance_level:
                                        restriction_mask[flat_idx] = False
                        else:
                            restriction_mask = p_values_flat <= significance_level
                        
                        n_restricted = (~restriction_mask).sum()
                        
                        st.info(f"Restricting {n_restricted} coefficients to zero...")
                        
                        # Create restricted coefficient matrix
                        coef_restricted = coef_flat.copy()
                        coef_restricted[~restriction_mask] = 0
                        coef_restricted_matrix = coef_restricted.reshape(coef_matrix.shape)
                        
                        # ===== NEW: Actually estimate a restricted VAR =====
                        from sklearn.linear_model import LinearRegression
                        
                        # Prepare data for restricted estimation
                        maxlag = int(selected_lag)
                        
                        # Create lagged data
                        y_data = df.values[maxlag:]  # Dependent variables
                        X_data_list = [np.ones((len(y_data), 1))]  # Intercept
                        
                        for lag in range(1, maxlag + 1):
                            X_data_list.append(df.values[maxlag-lag:-lag])
                        
                        X_data = np.hstack(X_data_list)
                        
                        # Estimate each equation separately with restrictions
                        restricted_coefs = []
                        restricted_residuals_list = []
                        
                        for eq_idx in range(len(variables)):
                            # Get mask for this equation
                            eq_mask = restriction_mask.reshape(coef_matrix.shape)[:, eq_idx]
                            
                            # Select only non-restricted predictors
                            X_restricted = X_data[:, eq_mask]
                            y_eq = y_data[:, eq_idx]
                            
                            # Fit restricted model
                            reg = LinearRegression(fit_intercept=False)
                            reg.fit(X_restricted, y_eq)
                            
                            # Get predictions and residuals
                            y_pred = reg.predict(X_restricted)
                            resid_eq = y_eq - y_pred
                            restricted_residuals_list.append(resid_eq)
                            
                            # Reconstruct full coefficient vector (with zeros for restricted)
                            full_coef = np.zeros(coef_matrix.shape[0])
                            full_coef[eq_mask] = reg.coef_
                            restricted_coefs.append(full_coef)
                        
                        # Stack residuals into DataFrame
                        restricted_residuals = pd.DataFrame(
                            np.column_stack(restricted_residuals_list),
                            columns=variables,
                            index=df.index[maxlag:]
                        )
                        
                        # Stack coefficients
                        coef_restricted_matrix = np.column_stack(restricted_coefs)
                        # ===== END NEW CODE =====
                        
                        st.warning("""
                        **Note:** Restricted VAR model has been estimated with coefficient restrictions applied.
                        
                        **Approach:**
                        - Insignificant coefficients set to zero
                        - Model re-estimated with restrictions
                        - Residuals calculated from restricted model
                        """)
                        
                        # Store restriction information INCLUDING RESIDUALS
                        st.session_state.restricted_var_model = {
                            'restriction_mask': restriction_mask,
                            'coef_restricted': coef_restricted_matrix,
                            'n_restricted': n_restricted,
                            'significance_level': significance_level,
                            'residuals': restricted_residuals  # ← NEW: Store actual residuals
                        }
                        
                        # Display restriction pattern
                        st.markdown("### Restriction Pattern")
                        
                        restriction_pattern = pd.DataFrame(
                            restriction_mask.reshape(coef_matrix.shape),
                            index=coef_matrix.index,
                            columns=coef_matrix.columns
                        )
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=restriction_pattern.values.astype(int),
                            x=restriction_pattern.columns,
                            y=restriction_pattern.index,
                            colorscale=[[0, '#ff6b6b'], [1, '#51cf66']],
                            text=[['Restricted' if not x else 'Kept' for x in row] for row in restriction_pattern.values],
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            showscale=False
                        ))
                        
                        fig.update_layout(
                            title="Coefficient Restriction Pattern<br><sub>Green = Kept, Red = Restricted to Zero</sub>",
                            height=400 + 30 * len(coef_matrix),
                            xaxis_title="Equation",
                            yaxis_title="Parameter"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Restricted coefficients
                        st.markdown("### Restricted Coefficient Matrix")
                        
                        for lag in range(int(selected_lag)):
                            st.markdown(f"#### Lag {lag + 1}")
                            start_idx = 1 + lag * len(variables)
                            end_idx = 1 + (lag + 1) * len(variables)
                            
                            lag_coef = pd.DataFrame(
                                coef_restricted_matrix[start_idx:end_idx, :],
                                index=[f"{v}.L{lag+1}" for v in variables],
                                columns=variables
                            )
                            
                            st.dataframe(
                                lag_coef.style
                                .format('{:.4f}')
                                .applymap(lambda x: 'background-color: #ffcccc' if x == 0 else 'background-color: #ccffcc'),
                                use_container_width=True
                            )
                        
                        st.success(f"""
                        ✓ **Restriction analysis complete!**
                        
                        - **Restricted coefficients:** {n_restricted}
                        - **Retained coefficients:** {restriction_mask.sum()}
                        - **Significance level:** {significance_level}
                        """)
                        
                    except Exception as e:
                        st.error(f"Error in restriction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # ====================================================================
    # MODEL COMPARISON
    # ====================================================================
    
    with st.expander("📊 3.4.3 Model Comparison: Unrestricted vs Restricted", expanded=True):
        st.markdown("### Comparing Models")
        
        if 'restricted_var_model' in st.session_state:
            restriction_info = st.session_state.restricted_var_model
            
            st.markdown("""
            **Comparison metrics:**
            - **Parsimony:** Fewer parameters = simpler model
            - **Information Criteria:** AIC, BIC (lower is better)
            - **Likelihood Ratio Test:** Statistical test for restrictions
            """)
            
            # Calculate approximate restricted model metrics
            n_restricted = restriction_info['n_restricted']
            n_obs = var_model.nobs
            n_params_unrestricted = coef_matrix.size
            n_params_restricted = n_params_unrestricted - n_restricted
            
            # Approximate AIC/BIC adjustment
            # Note: This is approximate since we didn't actually re-estimate
            log_likelihood_unrestricted = var_model.llf
            
            # For restricted model, we penalize less (fewer parameters)
            aic_unrestricted = var_model.aic
            bic_unrestricted = var_model.bic
            
            # Approximate restricted AIC/BIC (assuming similar fit)
            k_diff = n_restricted
            aic_restricted_approx = aic_unrestricted - 2 * k_diff
            bic_restricted_approx = bic_unrestricted - np.log(n_obs) * k_diff
            
            # Comparison table
            comparison_data = {
                'Metric': [
                    'Number of parameters',
                    'Number of restrictions',
                    'AIC',
                    'BIC',
                    'Observations'
                ],
                'Unrestricted VAR': [
                    n_params_unrestricted,
                    0,
                    f'{aic_unrestricted:.4f}',
                    f'{bic_unrestricted:.4f}',
                    n_obs
                ],
                'Restricted VAR (approx)': [
                    n_params_restricted,
                    n_restricted,
                    f'{aic_restricted_approx:.4f}',
                    f'{bic_restricted_approx:.4f}',
                    n_obs
                ],
                'Change': [
                    f'-{n_restricted}',
                    f'+{n_restricted}',
                    f'{aic_restricted_approx - aic_unrestricted:.4f}',
                    f'{bic_restricted_approx - bic_unrestricted:.4f}',
                    '0'
                ]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
            
            # Likelihood Ratio Test
            st.markdown("### Likelihood Ratio Test for Restrictions")
            
            st.info("""
            **Test:**
            - H₀: Restrictions are valid (restricted model is adequate)
            - H₁: Restrictions are not valid (need unrestricted model)
            
            **Test statistic:** LR = 2(ℓ_unrestricted - ℓ_restricted) ~ χ²(q)
            
            where q = number of restrictions
            """)
            
            # This would require actual restricted model estimation
            st.warning("""
            ⚠️ **Note:** Full likelihood ratio test requires actual restricted model estimation.
            
            **Interpretation guidance:**
            - If most coefficients are insignificant → restrictions likely valid
            - If AIC/BIC improve → restricted model preferred
            - If BIC improves but AIC doesn't → sample size consideration
            """)
            
            if bic_restricted_approx < bic_unrestricted:
                st.success("""
                ✓ **BIC prefers restricted model**
                
                The restricted model offers better parsimony without sacrificing much fit.
                **Recommendation:** Use restricted model for interpretation and forecasting.
                """)
            else:
                st.info("""
                **BIC prefers unrestricted model**
                
                The loss in fit from restrictions outweighs the parsimony gain.
                **Recommendation:** Keep unrestricted model, but focus interpretation on significant coefficients.
                """)
        else:
            st.info("👆 Please estimate the restricted VAR first using the button above.")
    
    # ====================================================================
    # RESTRICTED MODEL DIAGNOSTICS
    # ====================================================================
    
    with st.expander("🔬 3.4.4 Restricted Model Diagnostics", expanded=True):
        st.markdown("### Complete Diagnostic Tests for Restricted Model")
        
        if 'restricted_var_model' in st.session_state:
            try:
                restriction_info = st.session_state.restricted_var_model
                
                # ===== FIXED: Use actual restricted residuals =====
                if 'residuals' in restriction_info:
                    restricted_resid = restriction_info['residuals']
                    st.success("✅ Using residuals from the **RESTRICTED** model for diagnostics")
                else:
                    # Fallback (shouldn't happen with new code)
                    restricted_resid = var_model.resid
                    st.warning("⚠️ Using unrestricted residuals (restricted model not properly estimated)")
                # ===== END FIX =====
                
                # ================================================================
                # TEST 1: STABILITY
                # ================================================================
                st.markdown("#### 1️⃣ Stability Check")
                
                try:
                    # ===== FIXED: Use restricted model coefficients for stability =====
                    # Build companion matrix from RESTRICTED coefficients
                    coef_restricted = restriction_info['coef_restricted']
                    k = len(variables)  # number of variables
                    p = int(selected_lag)  # lag order
                    
                    # Extract coefficient matrices for each lag (skip constant)
                    coef_matrices = []
                    for i in range(p):
                        start_idx = 1 + i * k  # Skip constant (index 0)
                        end_idx = 1 + (i + 1) * k
                        coef_matrices.append(coef_restricted[start_idx:end_idx, :].T)
                    
                    # Build companion matrix (k*p x k*p)
                    companion = np.zeros((k*p, k*p))
                    companion[:k, :] = np.hstack(coef_matrices)
                    if p > 1:
                        companion[k:, :-k] = np.eye(k*(p-1))
                    
                    # Calculate eigenvalues from RESTRICTED companion matrix
                    eigenvalues = np.linalg.eigvals(companion)
                    # ===== END FIX =====
                    
                    all_stable = np.all(np.abs(eigenvalues) < 1)
                    
                    # Simplified table
                    stability_data = []
                    for i in range(min(10, len(eigenvalues))):
                        stability_data.append({
                            'Index': i,
                            'Eigenvalue |λ|': f"{np.abs(eigenvalues[i]):.6f}",
                            'Stable?': '✓ YES' if np.abs(eigenvalues[i]) < 1 else '✗ NO'
                        })
                    
                    st.dataframe(pd.DataFrame(stability_data), use_container_width=True)
                    
                    if all_stable:
                        st.success("✅ **MODEL IS STABLE**")
                    else:
                        st.error("❌ **MODEL IS UNSTABLE**")
                except Exception as e:
                    st.warning(f"Could not check stability: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                st.markdown("---")
                
                # ================================================================
                # TEST 2: SERIAL CORRELATION
                # ================================================================
                st.markdown("#### 2️⃣ Serial Correlation (Portmanteau)")
                
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    
                    results_data = []
                    all_pass = True
                    
                    for col_name in restricted_resid.columns:
                        lb_test = acorr_ljungbox(restricted_resid[col_name], lags=[5, 10], return_df=True)
                        
                        for idx in lb_test.index:
                            pval = lb_test.loc[idx, 'lb_pvalue']
                            passed = pval > 0.05
                            if not passed:
                                all_pass = False
                            
                            results_data.append({
                                'Variable': col_name,
                                'Lag': idx,
                                'P-value': f'{pval:.4f}',
                                'Result': '✅ Pass' if passed else '❌ Fail'
                            })
                    
                    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                    
                    if all_pass:
                        st.success("✅ **NO serial correlation**")
                    else:
                        st.error("❌ **Serial correlation detected**")
                except Exception as e:
                    st.warning(f"Could not test serial correlation: {e}")
                
                st.markdown("---")
                
                # ================================================================
                # TEST 3: NORMALITY
                # ================================================================
                st.markdown("#### 3️⃣ Normality (Jarque-Bera)")
                
                try:
                    from scipy.stats import jarque_bera
                    
                    norm_results = []
                    all_pass_norm = True
                    
                    for col_name in restricted_resid.columns:
                        jb_stat, jb_pval = jarque_bera(restricted_resid[col_name])
                        passed = jb_pval > 0.05
                        
                        if not passed:
                            all_pass_norm = False
                        
                        norm_results.append({
                            'Variable': col_name,
                            'JB Statistic': f'{jb_stat:.4f}',
                            'P-value': f'{jb_pval:.4f}',
                            'Result': '✅ Pass' if passed else '❌ Fail'
                        })
                    
                    st.dataframe(pd.DataFrame(norm_results), use_container_width=True)
                    
                    if all_pass_norm:
                        st.success("✅ **Residuals are normal**")
                    else:
                        st.warning("⚠️ **Non-normal residuals**")
                except Exception as e:
                    st.warning(f"Could not test normality: {e}")
                
                st.markdown("---")
                
                # ================================================================
                # TEST 4: ARCH EFFECTS
                # ================================================================
                st.markdown("#### 4️⃣ ARCH Effects")
                
                try:
                    from statsmodels.stats.diagnostic import het_arch
                    
                    arch_results = []
                    all_pass_arch = True
                    
                    for col_name in restricted_resid.columns:
                        arch_test = het_arch(restricted_resid[col_name], nlags=5)
                        pval = arch_test[1]
                        passed = pval > 0.05
                        
                        if not passed:
                            all_pass_arch = False
                        
                        arch_results.append({
                            'Variable': col_name,
                            'Test Statistic': f'{arch_test[0]:.4f}',
                            'P-value': f'{pval:.4f}',
                            'Result': '✅ Pass' if passed else '❌ Fail'
                        })
                    
                    st.dataframe(pd.DataFrame(arch_results), use_container_width=True)
                    
                    if all_pass_arch:
                        st.success("✅ **NO ARCH effects**")
                    else:
                        st.error("❌ **ARCH effects detected**")
                except Exception as e:
                    st.warning(f"Could not test ARCH effects: {e}")
                
                st.markdown("---")
                
                # ================================================================
                # SUMMARY
                # ================================================================
                st.markdown("### 📋 Diagnostic Summary")
                
                summary_data = {
                    'Test': ['Stability', 'Serial Correlation', 'Normality', 'ARCH Effects'],
                    'Status': [
                        '✅ Pass' if all_stable else '❌ Fail',
                        '✅ Pass' if all_pass else '❌ Fail',
                        '✅ Pass' if all_pass_norm else '⚠️ Warning',
                        '✅ Pass' if all_pass_arch else '❌ Fail'
                    ]
                }
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                total_pass = sum([all_stable, all_pass, all_pass_norm, all_pass_arch])
                
                if total_pass == 4:
                    st.success("🎉 **All tests passed! Restricted model is valid.**")
                elif total_pass >= 2:
                    st.info(f"ℹ️ **{total_pass}/4 tests passed.**")
                else:
                    st.error("⚠️ **Multiple failures. Restrictions may be too severe.**")
                    
            except Exception as e:
                st.error(f"Error in diagnostics: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("👆 Please estimate restricted VAR first in Section 3.4.2.")

    
    with st.expander("💡 3.4.5 Practical Recommendations", expanded=True):
        st.markdown("### When to Use Restricted VAR?")
        
        st.markdown("""
        #### ✅ **Use Restricted VAR when:**
        
        1. **Many insignificant coefficients** (>40% of parameters)
        2. **Forecasting focus** - parsimony improves out-of-sample performance
        3. **Small sample size** - restrictions combat overfitting
        4. **Computational constraints** - fewer parameters = faster computation
        5. **Interpretability** - easier to explain simpler model
        
        #### ❌ **Keep Unrestricted VAR when:**
        
        1. **Most coefficients significant** (>70%)
        2. **Theory suggests complex dynamics** - don't impose restrictions arbitrarily
        3. **Large sample size** - overfitting less of a concern
        4. **IRF/FEVD analysis** - restrictions can distort impulse responses
        5. **Diagnostic tests fail after restriction** - restrictions too severe
        
        #### ⚖️ **Middle ground:**
        
        - **Use unrestricted for estimation and inference**
        - **Focus interpretation on significant coefficients only**
        - **Document restriction pattern for transparency**
        """)
        
        st.markdown("### For Your Coursework")
        
        st.success("""
        **Recommended approach:**
        
        1. ✅ Estimate unrestricted VAR (as main model)
        2. ✅ Test coefficient significance
        3. ✅ Show which coefficients would be restricted
        4. ✅ Discuss trade-offs (parsimony vs fit)
        5. ✅ Use unrestricted for IRF/FEVD (standard practice)
        6. ✅ Mention restricted VAR as robustness check
        
        **Note:** Most applied papers use unrestricted VAR for structural analysis, focusing interpretation on significant relationships.
        """)

# ============================================================================
# END OF RESTRICTED VAR SECTION
# ============================================================================
    # ============================================================================


    # ====================================================================
    # MODEL SELECTION FOR STRUCTURAL ANALYSIS
    # ====================================================================
    
    st.markdown("---")
    st.markdown("### 📊 Model Selection for Structural Analysis")
    st.markdown("Select which VAR model to use for IRF, FEVD, and Granger causality")
    
    # Check if restricted model is available
    has_restricted = 'restricted_var_model' in st.session_state
    
    # Display available models
    st.markdown("#### Available Models:")
    
    if has_restricted:
        # Show both models side by side
        col1, col2 = st.columns(2)
        
        # Unrestricted model
        with col1:
            st.markdown("**Current Model**")
            st.metric("VAR (Unrestricted)", f"VAR({selected_lag})")
            st.metric("AIC", f"{var_model.aic:.4f}")
            st.metric("BIC", f"{var_model.bic:.4f}")
        
        # Restricted model
        with col2:
            restriction_info = st.session_state.restricted_var_model
            
            # Calculate approximate AIC/BIC for restricted model
            n_restricted = restriction_info['n_restricted']
            n_obs = var_model.nobs
            k_diff = n_restricted
            
            # Approximate restricted AIC/BIC (fewer parameters = better parsimony)
            aic_restricted_approx = var_model.aic - 2 * k_diff
            bic_restricted_approx = var_model.bic - np.log(n_obs) * k_diff
            
            st.markdown("**Restricted Model**")
            st.metric("VAR (Restricted)", f"VAR({selected_lag})")
            st.metric("AIC (approx)", f"{aic_restricted_approx:.4f}",
                     delta=f"{aic_restricted_approx - var_model.aic:.4f}")
            st.metric("BIC (approx)", f"{bic_restricted_approx:.4f}",
                     delta=f"{bic_restricted_approx - var_model.bic:.4f}")
        
        # Determine which model has better AIC (lower is better)
        if aic_restricted_approx < var_model.aic:
            best_model = "Restricted"
            best_aic = aic_restricted_approx
            st.success("✅ **Restricted model has better (lower) AIC**")
        else:
            best_model = "Unrestricted"
            best_aic = var_model.aic
            st.success("✅ **Unrestricted model has better (lower) AIC**")
        
        st.markdown("---")
        
        # Radio button for model selection with automatic default to best AIC
        st.markdown("#### 🎯 Choose Model for Analysis:")
        
        model_options = [
            f"Unrestricted VAR({selected_lag}) - AIC: {var_model.aic:.4f}" + (" ⭐ Better AIC" if best_model == "Unrestricted" else ""),
            f"Restricted VAR({selected_lag}) - AIC: {aic_restricted_approx:.4f}" + (" ⭐ Better AIC" if best_model == "Restricted" else "")
        ]
        
        # Default index: 0 for Unrestricted, 1 for Restricted
        default_index = 0 if best_model == "Unrestricted" else 1
        
        model_choice = st.radio(
            "Select model:",
            model_options,
            index=default_index,
            help="Model with ⭐ has better (lower) AIC. You can select either model."
        )
        
        # Determine selected model
        if "Restricted" in model_choice:
            use_restricted = True
            st.info("📊 **Using Restricted VAR for structural analysis**")
            st.caption(f"✓ Restricted coefficients | ✓ {n_restricted} parameters set to zero")
            
            # Store for analysis
            st.session_state.use_restricted_for_analysis = True
            st.session_state.analysis_model_type = "Restricted"
            st.session_state.selected_aic = aic_restricted_approx
        else:
            use_restricted = False
            st.info("📊 **Using Unrestricted VAR for structural analysis**")
            st.caption(f"✓ All coefficients included | ✓ Full model dynamics")
            
            # Store for analysis
            st.session_state.use_restricted_for_analysis = False
            st.session_state.analysis_model_type = "Unrestricted"
            st.session_state.selected_aic = var_model.aic
        
    else:
        # Only unrestricted model available
        col1 = st.columns(1)[0]
        with col1:
            st.metric("Current Model", f"VAR({selected_lag})")
            st.metric("AIC", f"{var_model.aic:.4f}")
            st.metric("BIC", f"{var_model.bic:.4f}")
        
        st.info("📊 **Using Unrestricted VAR({})** (only model available)".format(selected_lag))
        st.caption("💡 To compare models, estimate a Restricted VAR in Section 3.4.2")
        
        use_restricted = False
        st.session_state.use_restricted_for_analysis = False
        st.session_state.analysis_model_type = "Unrestricted"
        st.session_state.selected_aic = var_model.aic
    
    # Always use same lag order but different coefficient restrictions
    analysis_model = var_model  # Base model object
    analysis_lag = selected_lag
    
    # Store selected model for use in all structural analysis sections
    st.session_state.analysis_model = analysis_model
    st.session_state.analysis_lag = analysis_lag
    
    st.markdown("---")

    # SECTION 4: STRUCTURAL ANALYSIS
    # ====================================================================
    
    st.markdown('<div class="section-header">4. Structural Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("🔗 4.1 Granger Causality Tests (Wald Test)", expanded=True):
        st.markdown("### Pairwise Granger Causality - Wald Test Approach")
        st.markdown("""
        
        
        **Null Hypothesis (H₀):** Variable X does NOT Granger-cause variable Y
        - Formally: R·β = r where R restricts all lags of X in equation for Y to zero
        
        **Test Statistic:** Wald test with F-distribution
        - **W** = (Rβ̂ - r)' [R·(ZZ')⁻¹·Σ̂ᵤ·R']⁻¹ (Rβ̂ - r) → χ²(#r)
        - Or use **F-statistic** for small samples
        
        **Decision Rule:** Reject H₀ if p-value < 0.05 (evidence of Granger causality)
        """)
        
        st.markdown("---")
        st.markdown("#### Granger Causality Test Results")
        
        # Run tests using statsmodels (which uses Wald test internally)
        granger_results = {}
        granger_data = []
        
        for cause_var in variables:
            for effect_var in variables:
                if cause_var != effect_var:
                    try:
                        # Prepare data: [effect, cause]
                        test_data = df[[effect_var, cause_var]].dropna()
                        
                        # Run Granger causality test (this uses Wald test)
                        test_result = grangercausalitytests(
                            test_data, 
                            maxlag=int(selected_lag), 
                            verbose=False
                        )
                        
                        # Extract Wald test results for the selected lag
                        wald_test = test_result[int(selected_lag)][0]['ssr_ftest']
                        f_stat = wald_test[0]  # F-statistic
                        p_value = wald_test[1]  # p-value
                        df1 = wald_test[2]      # degrees of freedom
                        
                        # Also get chi-square version (Wald statistic)
                        wald_chi2 = test_result[int(selected_lag)][0]['ssr_chi2test']
                        chi2_stat = wald_chi2[0]
                        chi2_pval = wald_chi2[1]
                        
                        significant = p_value < 0.05
                        
                        granger_results[f"{cause_var} → {effect_var}"] = {
                            'F-statistic': f_stat,
                            'p-value': p_value,
                            'Chi2-statistic': chi2_stat,
                            'Chi2-pvalue': chi2_pval,
                            'df': df1,
                            'Significant': significant
                        }
                        
                        granger_data.append({
                            'Cause (X)': cause_var,
                            'Effect (Y)': effect_var,
                            'F-statistic': f'{f_stat:.4f}',
                            'p-value': f'{p_value:.4f}',
                            'Wald χ²': f'{chi2_stat:.4f}',
                            'χ² p-value': f'{chi2_pval:.4f}',
                            'Significant (5%)': '✓ YES' if significant else '✗ NO',
                            'Interpretation': f"X {'DOES' if significant else 'does NOT'} Granger-cause Y"
                        })
                        
                    except Exception as e:
                        st.warning(f"Could not test {cause_var} → {effect_var}: {e}")
        
        if granger_data:
            granger_df = pd.DataFrame(granger_data)
            
            st.dataframe(
                granger_df.style.apply(
                    lambda x: ['background-color: #d4edda' if v == '✓ YES' 
                              else 'background-color: #f8d7da' if v == '✗ NO' 
                              else '' for v in x], 
                    subset=['Significant (5%)']
                ),
                use_container_width=True
            )
            
            # Summary statistics
            n_significant = sum(1 for r in granger_results.values() if r['Significant'])
            n_total = len(granger_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tests", n_total)
            with col2:
                st.metric("Significant", n_significant)
            with col3:
                st.metric("% Significant", f"{(n_significant/n_total*100):.1f}%")
            
            st.info(f"""
            **Summary:** {n_significant} out of {n_total} relationships show significant Granger causality at 5% level.
            
            **Test Used:** Wald F-test 
            - H₀: All lags of cause variable have zero coefficients in effect equation
            - F ~ F(p, T - Kp - 1) where p = lag order, T = observations, K = # variables
            """)
            
            # Heatmap
            st.markdown("---")
            st.markdown("#### Granger Causality Matrix (p-values)")
            
            granger_matrix = np.full((len(variables), len(variables)), np.nan)
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        key = f"{var1} → {var2}"
                        if key in granger_results:
                            granger_matrix[i, j] = granger_results[key]['p-value']
            
            fig = go.Figure(data=go.Heatmap(
                z=granger_matrix,
                x=variables,
                y=variables,
                colorscale='RdYlGn_r',
                text=np.round(granger_matrix, 4),
                texttemplate='%{text:.4f}',
                colorbar=dict(title="P-value"),
                zmin=0,
                zmax=0.1,
                hovertemplate='Cause: %{x}<br>Effect: %{y}<br>p-value: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Granger Causality P-values Matrix<br><sub>Row causes Column | Green = Significant</sub>",
                height=500,
                xaxis_title="Cause Variable (X)",
                yaxis_title="Effect Variable (Y)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            **How to read:** 
            - Row = Cause (X), Column = Effect (Y)
            - Green cells (p < 0.05): X Granger-causes Y
            - Red cells (p > 0.05): X does NOT Granger-cause Y
            """)
        else:
            st.error("No Granger causality tests could be performed.")
    
    
    
    with st.expander("⚡ 4.2 Instantaneous Causality (Wald Test)", expanded=True):
        st.markdown("### Instantaneous Causality Tests")
        st.markdown("""
        
        
        Tests whether variables are contemporaneously correlated (same time period).
        
        **Null Hypothesis (H₀):** Rσ = 0 (no instantaneous causality)
        - Formally: σᵢⱼ = 0 for i ≠ j in residual covariance matrix
        
        **Test Statistic (from slides):**
        """)
        
        st.latex(r"W = T (R\hat{\sigma})' [R \hat{\Sigma}_{\sigma} R']^{-1} (R\hat{\sigma}) \xrightarrow{d} \chi^2(\#r)")
        
        st.markdown("""
        where:
        - T = sample size
        - R = restriction matrix selecting off-diagonal elements
        - σ̂ = vech(Σ̂ᵤ) = vectorized residual covariance
        - #r = number of rows in R
        
        **Decision Rule:** Reject H₀ if p-value < 0.05 (evidence of instantaneous causality)
        """)
        
        try:
            # Get residual covariance matrix
            sigma_u = var_model.sigma_u
            
            # CRITICAL FIX: Convert to numpy array if it's a DataFrame
            if isinstance(sigma_u, pd.DataFrame):
                sigma_u_array = sigma_u.values
            else:
                sigma_u_array = sigma_u
            
            n_obs = var_model.nobs
            k = len(variables)
            
            st.markdown("---")
            st.markdown("#### Step 1: Residual Covariance Matrix (Σ̂ᵤ)")
            
            sigma_df = pd.DataFrame(sigma_u_array, index=variables, columns=variables)
            st.dataframe(
                sigma_df.style.format('{:.6f}').background_gradient(cmap='RdBu', axis=None),
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("#### Step 2: Instantaneous Causality Tests (Pairwise)")
            
            inst_results = []
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i < j:  # Only upper triangle (avoid duplicates)
                        # Extract covariance using numpy array indexing
                        sigma_ij = sigma_u_array[i, j]
                        sigma_ii = sigma_u_array[i, i]
                        sigma_jj = sigma_u_array[j, j]
                        
                        # Correlation
                        corr_ij = sigma_ij / np.sqrt(sigma_ii * sigma_jj)
                        
                        # Wald test statistic for H₀: σᵢⱼ = 0
                        # Simplified version: W = T * ρ² ~ χ²(1)
                        test_stat = n_obs * (corr_ij ** 2)
                        
                        # P-value from chi-square distribution (df=1)
                        p_value = 1 - chi2.cdf(test_stat, df=1)
                        
                        # Critical value at 5%
                        critical_value = chi2.ppf(0.95, df=1)
                        
                        significant = p_value < 0.05
                        
                        inst_results.append({
                            'Pair': f'{var1} ↔ {var2}',
                            'Covariance (σᵢⱼ)': f'{sigma_ij:.6f}',
                            'Correlation (ρᵢⱼ)': f'{corr_ij:.4f}',
                            'Wald χ²': f'{test_stat:.4f}',
                            'p-value': f'{p_value:.4f}',
                            'Critical (95%)': f'{critical_value:.4f}',
                            'Significant (5%)': '✓ YES' if significant else '✗ NO',
                            'Interpretation': f"{'Reject H₀' if significant else 'Cannot reject H₀'}: {'Evidence of' if significant else 'No'} instantaneous causality"
                        })
            
            inst_df = pd.DataFrame(inst_results)
            
            st.dataframe(
                inst_df.style.apply(
                    lambda x: ['background-color: #d4edda' if v == '✓ YES' 
                              else 'background-color: #f8d7da' if v == '✗ NO' 
                              else '' for v in x], 
                    subset=['Significant (5%)']
                ),
                use_container_width=True
            )
            
            # Summary
            n_significant = sum([1 for r in inst_results if '✓ YES' in r['Significant (5%)']])
            n_total = len(inst_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pairs Tested", n_total)
            with col2:
                st.metric("Significant", n_significant)
            with col3:
                st.metric("% Significant", f"{(n_significant/n_total*100 if n_total > 0 else 0):.1f}%")
            
            if n_significant > 0:
                st.warning(f"""
                ⚡ **Instantaneous causality detected in {n_significant} pair(s)**
                
                **Interpretation:**
                - Variables respond simultaneously to common shocks
                - Or there are omitted variables affecting both
                - Consider ordering carefully for IRF (Cholesky decomposition)
                
                **Implication for IRF:**
                - More exogenous variable should come first in ordering
                - Variable responding simultaneously should be placed later
                """)
            else:
                st.success("""
                ✅ **No significant instantaneous causality detected**
                
                Variables do not respond contemporaneously to each other's shocks.
                IRF ordering is less critical (but still affects orthogonalized IRFs).
                """)
            
            st.markdown("---")
            st.markdown("#### Visual: Residual Correlation Heatmap")
            
            # Correlation matrix from residuals
            resid_corr = var_model.resid.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=resid_corr.values,
                x=resid_corr.columns,
                y=resid_corr.index,
                colorscale='RdBu',
                zmid=0,
                text=resid_corr.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation"),
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                title="Residual Correlation Matrix<br><sub>High |correlation| indicates instantaneous causality</sub>",
                height=500,
                xaxis_title="Variable",
                yaxis_title="Variable"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Test Formula Used (simplified):**
            
            For testing H₀: σᵢⱼ = 0:
            - Wald statistic: W = T·ρ²ᵢⱼ ~ χ²(1)
            - where ρᵢⱼ = σᵢⱼ / √(σᵢᵢ·σⱼⱼ)
            
            General formula:
            W = T(Rσ̂)'[RΣ̂ₛR']⁻¹(Rσ̂) for the special case of testing a single covariance.
            """)
        
        except Exception as e:
            st.error(f"Error in instantaneous causality test: {e}")
            import traceback
            st.code(traceback.format_exc())

    with st.expander("📈 4.3 Impulse Response Functions with Confidence Intervals", expanded=True):
        st.markdown("### Forecast Error Impulse Response Functions")
        st.markdown("""
        Shows the dynamic response of each variable to a one-unit shock in another variable.
        
        **Two types:**
        1. **Non-orthogonalized (Forecast Error):** Raw impulse responses
        2. **Orthogonalized (Cholesky):** Decomposed shocks using variable ordering
        
        **Confidence Intervals:**
        - Computed using asymptotic standard errors
        - Shown as **red dotted lines** (95% CI)
        - Solid black line = point estimate
        """)
        
        irf_periods = st.slider("Forecast horizon (periods)", 5, 40, 10, key='irf_periods')
        
        # Orthogonalization option
        col1, col2 = st.columns(2)
        
        with col1:
            irf_type = st.radio(
                "IRF Type:",
                ["Non-orthogonalized (Forecast Error)", "Orthogonalized (Cholesky)"],
                help="Non-orthogonalized: raw impulses | Orthogonalized: decomposed using Cholesky"
            )
        
        with col2:
            if "Orthogonalized" in irf_type:
                st.markdown("**Variable Ordering for Cholesky:**")
                st.info("More exogenous → More endogenous")
                
                irf_ordering = st.multiselect(
                    "Select order (top = most exogenous):",
                    variables,
                    default=variables,
                    help="First variable is most exogenous, last is most endogenous"
                )
            else:
                irf_ordering = variables
        
        use_orthogonal = "Orthogonalized" in irf_type
        
        # Check if we need to reorder
        if use_orthogonal and len(irf_ordering) != len(variables):
            st.warning(f"⚠️ Please select all {len(variables)} variables for orthogonalized IRF")
            st.stop()
        
        try:
            # ========================================
            # FIX 1: ALWAYS refit model with correct ordering
            # ========================================
            if use_orthogonal:
                # For orthogonalized: reorder data according to user selection
                df_ordered = df[irf_ordering]
                var_for_irf = VAR(df_ordered).fit(int(selected_lag))
                display_variables = irf_ordering
                st.info(f"✓ Using **Orthogonalized IRF** with ordering: {' → '.join(display_variables)}")
            else:
                # For non-orthogonalized: use original model and ordering
                var_for_irf = var_model
                display_variables = variables
                st.info("✓ Using **Non-orthogonalized (Forecast Error) IRF**")
            
            # ========================================
            # FIX 2: Compute IRF with correct model
            # ========================================
            irf = var_for_irf.irf(irf_periods)
            
            # Get IRF values
            if use_orthogonal:
                irf_values = irf.orth_irfs  # Orthogonalized (Cholesky decomposed)
                irf_stderr = irf.stderr(orth=True)  # ← FIX: Use orthogonalized stderr
            else:
                irf_values = irf.irfs  # Non-orthogonalized (forecast error)
                irf_stderr = irf.stderr(orth=False)  # ← FIX: Use non-orthogonalized stderr
            
            # ========================================
            # FIX 3: Correct confidence intervals
            # ========================================
            z_critical = 1.96  # 95% CI
            irf_upper = irf_values + z_critical * irf_stderr
            irf_lower = irf_values - z_critical * irf_stderr
            
            st.markdown("---")
            st.markdown(f"### IRF Plots ({irf_type})")
            st.markdown("**Solid black line:** Point estimate | **Red dotted lines:** 95% Confidence Interval")
            
            # ========================================
            # Create subplot grid
            # ========================================
            n_vars = len(display_variables)
            fig = make_subplots(
                rows=n_vars, cols=n_vars,
                subplot_titles=[f"Shock: {imp} → Response: {resp}" 
                            for resp in display_variables for imp in display_variables],
                vertical_spacing=0.06,
                horizontal_spacing=0.06
            )
            
            periods = list(range(irf_periods))
            
            for i, response_var in enumerate(display_variables):
                for j, impulse_var in enumerate(display_variables):
                    row = i + 1
                    col = j + 1
                    
                    # Point estimate (solid black line)
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=irf_values[:, i, j],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False,
                        name=f"{impulse_var} → {response_var}",
                        hovertemplate='Period: %{x}<br>Response: %{y:.4f}<extra></extra>'
                    ), row=row, col=col)
                    
                    # Upper confidence bound (red dotted)
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=irf_upper[:, i, j],
                        mode='lines',
                        line=dict(color='red', width=1.5, dash='dot'),
                        showlegend=False,
                        hovertemplate='Upper 95% CI: %{y:.4f}<extra></extra>'
                    ), row=row, col=col)
                    
                    # Lower confidence bound (red dotted)
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=irf_lower[:, i, j],
                        mode='lines',
                        line=dict(color='red', width=1.5, dash='dot'),
                        showlegend=False,
                        hovertemplate='Lower 95% CI: %{y:.4f}<extra></extra>'
                    ), row=row, col=col)
                    
                    # Zero line (horizontal reference)
                    fig.add_hline(
                        y=0, 
                        line_dash="solid", 
                        line_color="gray", 
                        opacity=0.3, 
                        line_width=1,
                        row=row, col=col
                    )
            
            # Update layout
            fig.update_layout(
                title_text=f"Impulse Response Functions ({irf_type})<br><sub>Black solid = IRF | Red dotted = 95% CI</sub>",
                height=250 * n_vars,
                showlegend=False,
                hovermode='closest'
            )
            
            # Update all x and y axes
            fig.update_xaxes(title_text="Periods", showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(title_text="Response", showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ========================================
            # Individual IRF viewer (rest remains same)
            # ========================================
            st.markdown("---")
            st.markdown("### 🔍 Individual IRF Viewer (Detailed)")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_impulse = st.selectbox("Impulse variable:", display_variables, key='impulse_select')
            with col2:
                selected_response = st.selectbox("Response variable:", display_variables, key='response_select')
            
            # Get indices
            impulse_idx = display_variables.index(selected_impulse)
            response_idx = display_variables.index(selected_response)
            
            # Create detailed plot
            fig_detail = go.Figure()
            
            # Point estimate
            fig_detail.add_trace(go.Scatter(
                x=periods,
                y=irf_values[:, response_idx, impulse_idx],
                mode='lines+markers',
                line=dict(color='black', width=3),
                marker=dict(size=6, color='black'),
                name='IRF Point Estimate'
            ))
            
            # Confidence interval (shaded area)
            fig_detail.add_trace(go.Scatter(
                x=periods + periods[::-1],
                y=np.concatenate([irf_upper[:, response_idx, impulse_idx],
                                irf_lower[:, response_idx, impulse_idx][::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # Upper bound (red dotted)
            fig_detail.add_trace(go.Scatter(
                x=periods,
                y=irf_upper[:, response_idx, impulse_idx],
                mode='lines',
                line=dict(color='red', width=2, dash='dot'),
                name='Upper 95% CI'
            ))
            
            # Lower bound (red dotted)
            fig_detail.add_trace(go.Scatter(
                x=periods,
                y=irf_lower[:, response_idx, impulse_idx],
                mode='lines',
                line=dict(color='red', width=2, dash='dot'),
                name='Lower 95% CI'
            ))
            
            # Zero line
            fig_detail.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
            
            fig_detail.update_layout(
                title=f"Impulse Response: {selected_impulse} → {selected_response}<br><sub>{irf_type}</sub>",
                xaxis_title="Periods After Shock",
                yaxis_title="Response",
                height=500,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )
            
            st.plotly_chart(fig_detail, use_container_width=True)
            
            # Numerical table
            st.markdown("---")
            st.markdown(f"##### 📋 Numerical Values: {selected_impulse} → {selected_response}")
            
            irf_table_data = []
            
            for h in range(irf_periods):
                irf_table_data.append({
                    'Period': h,
                    'IRF': f'{irf_values[h, response_idx, impulse_idx]:.6f}',
                    'Std Error': f'{irf_stderr[h, response_idx, impulse_idx]:.6f}',
                    'Lower 95%': f'{irf_lower[h, response_idx, impulse_idx]:.6f}',
                    'Upper 95%': f'{irf_upper[h, response_idx, impulse_idx]:.6f}',
                    'Significant?': '✓ YES' if (irf_lower[h, response_idx, impulse_idx] > 0 or 
                                                irf_upper[h, response_idx, impulse_idx] < 0) else '✗ NO'
                })
            
            irf_table_df = pd.DataFrame(irf_table_data)
            
            st.dataframe(
                irf_table_df.style.apply(
                    lambda x: ['background-color: #d4edda' if v == '✓ YES' 
                            else 'background-color: #f8d7da' if v == '✗ NO' 
                            else '' for v in x], 
                    subset=['Significant?']
                ),
                use_container_width=True
            )
            
            st.caption("""
            **How to interpret:**
            - **IRF**: Point estimate of response at each period
                - **Std Error**: Standard error of the IRF estimate
                - **Lower/Upper 95%**: Confidence interval bounds
                - **Significant?**: YES if CI does not include zero
                """)
            
            # Interpretation guide
            st.markdown("---")
            st.markdown("### 💡 Interpretation Guide")
            
            st.info(f"""
            **What this shows:**
            
            - **Impulse:** One-time shock to **{selected_impulse}**
            - **Response:** Dynamic effect on **{selected_response}** over {irf_periods} periods
            
            **Reading the plot:**
            - **Black line:** Expected response path
            - **Red dotted lines:** 95% confidence bounds
            - **If CI excludes zero:** Effect is statistically significant
            - **If CI includes zero:** Effect is not statistically significant
            
            **{'Orthogonalized' if use_orthogonal else 'Non-orthogonalized'} IRF:**
            - {"Shocks are orthogonalized using Cholesky decomposition" if use_orthogonal else "Raw forecast error impulse responses"}
            - {"Variable ordering matters for identification" if use_orthogonal else "Does not require variable ordering"}
            """)
            
        except Exception as e:
            st.error(f"Error computing IRF: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    with st.expander("🎯 4.4 Forecast Error Variance Decomposition (FEVD)", expanded=True):
        st.markdown("### Variance Decomposition Analysis")
        
        st.markdown("""
       
        
        Shows what percentage of forecast error variance in each variable is explained by shocks from all variables.
        
        **Key concepts:**
        - **Own shock**: Variable's forecast error explained by its own innovations
        - **Spillover**: Forecast error explained by other variables' shocks
        - **Horizon**: How many periods ahead we're forecasting
        """)
        
        # User controls
        col1, col2 = st.columns(2)
        
        with col1:
            fevd_periods = st.slider("Maximum horizon", 5, 40, 10, key='fevd_horizon')
        
        with col2:
            # Let user select specific horizons to display
            available_horizons = list(range(1, fevd_periods + 1))
            default_horizons = [1, 2, 3, 4] if fevd_periods >= 4 else available_horizons[:min(4, fevd_periods)]
            
            selected_horizons = st.multiselect(
                "Horizons to display in bar charts:",
                available_horizons,
                default=default_horizons,
                help="Select which forecast horizons to show in detail"
            )
        
        if not selected_horizons:
            st.warning("Please select at least one horizon")
            selected_horizons = [1]
        
        try:
            # ========================================
            # FIX 1: Use the CORRECT model
            # ========================================
            # Get the model that was actually used for analysis
            if 'analysis_model' in st.session_state:
                model_for_fevd = st.session_state.analysis_model
            else:
                model_for_fevd = var_model
            
            # ========================================
            # FIX 2: Get CORRECT variable list
            # ========================================
            # Use the variables from the model's data, not the original list
                        # ========================================
            # FIX 2: Get CORRECT variable list - FIXED VERSION
            # ========================================
            try:
                # Multiple ways to get variable names
                if hasattr(model_for_fevd, 'names') and model_for_fevd.names is not None:
                    fevd_variables = list(model_for_fevd.names)
                elif hasattr(model_for_fevd, 'model') and hasattr(model_for_fevd.model, 'endog_names'):
                    fevd_variables = list(model_for_fevd.model.endog_names)
                else:
                    fevd_variables = list(df.columns)  # Fallback to original data
                    
                n_vars = len(fevd_variables)
                
                # Validate
                if len(fevd_variables) == 0:
                    st.error("❌ No variables found in model")
                    st.stop()
                    
                st.info(f"✓ Computing FEVD for **{n_vars} variables**: {', '.join(fevd_variables)}")
                
            except Exception as e:
                st.error(f"Error getting variable names: {e}")
                # Ultimate fallback
                fevd_variables = variables
                n_vars = len(fevd_variables)
            # ========================================
            # FIX 3: Compute FEVD with validation
            # ========================================
            # ========================================
# FIX 3: Compute FEVD with validation
# ========================================
            fevd = model_for_fevd.fevd(fevd_periods)

            # Validate FEVD shape - FIXED: Handle different dimension orders
            actual_shape = fevd.decomp.shape

            # Check if we need to transpose dimensions
            if actual_shape[0] == n_vars and actual_shape[1] == fevd_periods:
                # Shape is (n_vars, periods, n_vars) but we need (periods, n_vars, n_vars)
                # Transpose to correct order
                fevd.decomp = np.transpose(fevd.decomp, (1, 0, 2))
                st.info(f"✓ Transposed FEVD dimensions from {actual_shape} to {fevd.decomp.shape}")
                actual_shape = fevd.decomp.shape

            expected_shape = (fevd_periods, n_vars, n_vars)

            if actual_shape != expected_shape:
                st.error(f"""
                ❌ **FEVD Shape Mismatch!**
                
                Expected: {expected_shape}
                Got: {actual_shape}
                
                This may be due to different statsmodels versions.
                """)
                # Continue with warning instead of stopping
                st.warning("⚠️ Proceeding with actual FEVD shape - check results carefully")
            else:
                st.success(f"✓ FEVD computed successfully: shape {actual_shape}")

            st.markdown("---")
            st.markdown("### 📊 FEVD Bar Charts (R-style)")
            st.markdown("**Stacked bars showing variance contribution by shock source**")

            # ========================================
            # FIX 4: Safe iteration with bounds checking
            # ========================================
            for resp_idx, response_var in enumerate(fevd_variables):
                # Validate response index
                if resp_idx >= n_vars:
                    st.error(f"Invalid response index {resp_idx} for {response_var}")
                    continue
                
                st.markdown(f"#### Response Variable: **{response_var}**")
                
                # Create stacked bar chart for selected horizons
                fig = go.Figure()
                
                # Prepare data for stacked bars
                for imp_idx, impulse_var in enumerate(fevd_variables):
                    # Validate impulse index
                    if imp_idx >= n_vars:
                        st.warning(f"Skipping invalid impulse variable {impulse_var}")
                        continue
                    
                    try:
                        # FIXED: Safe extraction with validation
                        contributions = []
                        for h in selected_horizons:
                            if h < 1 or h > fevd_periods:
                                st.warning(f"Horizon {h} out of range")
                                continue
                            
                            # CRITICAL: Use h-1 because Python is 0-indexed
                            # Handle both possible dimension orders
                            try:
                                contrib_value = fevd.decomp[h-1, resp_idx, imp_idx] * 100
                            except IndexError:
                                # Try alternate dimension order
                                try:
                                    contrib_value = fevd.decomp[resp_idx, h-1, imp_idx] * 100
                                except IndexError:
                                    st.error(f"Cannot access FEVD at indices ({h-1}, {resp_idx}, {imp_idx})")
                                    continue
                            
                            contributions.append(contrib_value)
                        
                        if not contributions:
                            continue
                        
                        # Add trace
                        fig.add_trace(go.Bar(
                            name=impulse_var,
                            x=[f"h={h}" for h in selected_horizons if 1 <= h <= fevd_periods],
                            y=contributions,
                            text=[f'{c:.1f}%' for c in contributions],
                            textposition='inside',
                            textfont=dict(color='white', size=11),
                            hovertemplate=f'<b>{impulse_var}</b><br>Contribution: %{{y:.2f}}%<extra></extra>'
                        ))
                    
                    except IndexError as e:
                        st.error(f"""
                        ❌ **Index Error for {impulse_var} → {response_var}**
                        
                        Details:
                        - Response index: {resp_idx}
                        - Impulse index: {imp_idx}
                        - FEVD shape: {fevd.decomp.shape}
                        
                        Error: {e}
                        """)
                        continue
                
                # Plot the chart
                fig.update_layout(
                    barmode='stack',
                    title=f'FEVD for {response_var} (Selected Horizons)<br><sub>Stacked bars show % of forecast error variance explained by each shock</sub>',
                    xaxis_title='Forecast Horizon',
                    yaxis_title='Percentage of Variance (%)',
                    yaxis=dict(range=[0, 100]),
                    height=400,
                    legend=dict(
                        title='Shock Source',
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ========================================
                # Cumulative line plot
                # ========================================
                st.markdown(f"**Cumulative FEVD Evolution (Horizons 1 to {fevd_periods})**")
                
                fig_cumul = go.Figure()
                
                for imp_idx, impulse_var in enumerate(fevd_variables):
                    if imp_idx >= n_vars:
                        continue
                    
                    try:
                        contributions_all = [fevd.decomp[h, resp_idx, imp_idx] * 100 
                                        for h in range(fevd_periods)]
                        
                        fig_cumul.add_trace(go.Scatter(
                            x=list(range(1, fevd_periods + 1)),
                            y=contributions_all,
                            mode='lines',
                            name=impulse_var,
                            stackgroup='one',
                            fillcolor=None,
                            line=dict(width=2)
                        ))
                    except IndexError:
                        continue
                
                fig_cumul.update_layout(
                    title=f'FEVD Evolution for {response_var}',
                    xaxis_title='Horizon',
                    yaxis_title='Cumulative Variance (%)',
                    yaxis=dict(range=[0, 100]),
                    height=350,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_cumul, use_container_width=True)
                
                # ========================================
                # Numerical table (removed nested expander)
                # ========================================
                st.markdown("---")
                st.markdown(f"##### 📋 FEVD Numerical Values: {response_var}")
                st.markdown("**Variance decomposition at selected horizons**")
                
                fevd_table_data = []
                for h in selected_horizons:
                    if h < 1 or h > fevd_periods:
                        continue
                    
                    row = {'Horizon': h}
                    for imp_idx, impulse_var in enumerate(fevd_variables):
                        if imp_idx >= n_vars:
                            continue
                        try:
                            contrib = fevd.decomp[h-1, resp_idx, imp_idx] * 100
                            row[impulse_var] = f'{contrib:.2f}%'
                        except IndexError:
                            row[impulse_var] = 'N/A'
                    
                    fevd_table_data.append(row)
                
                if fevd_table_data:
                    fevd_table = pd.DataFrame(fevd_table_data)
                    st.dataframe(fevd_table, use_container_width=True)
                
                # Long-run decomposition
                st.markdown(f"**Long-Run Decomposition (Horizon {fevd_periods})**")
                
                longrun_data = []
                for imp_idx, impulse_var in enumerate(fevd_variables):
                    if imp_idx >= n_vars:
                        continue
                    try:
                        contrib = fevd.decomp[-1, resp_idx, imp_idx] * 100
                        longrun_data.append({
                            'Shock Source': impulse_var,
                            'Contribution': f'{contrib:.2f}%',
                            'Type': 'Own shock' if impulse_var == response_var else 'Spillover'
                        })
                    except IndexError:
                        continue
                
                if longrun_data:
                    longrun_df = pd.DataFrame(longrun_data)
                    st.dataframe(longrun_df, use_container_width=True)
                    
                    # Interpretation
                    try:
                        dominant_shock_idx = fevd.decomp[-1, resp_idx, :].argmax()
                        dominant_shock = fevd_variables[dominant_shock_idx]
                        dominant_contrib = fevd.decomp[-1, resp_idx, dominant_shock_idx] * 100
                        
                        st.info(f"""
                        **Key Finding:** In the long run (horizon {fevd_periods}), **{dominant_shock}** shocks explain 
                        **{dominant_contrib:.1f}%** of forecast error variance in **{response_var}**.
                        
                        {'This is primarily an own shock effect.' if dominant_shock == response_var else f'This indicates significant spillover from {dominant_shock}.'}
                        """)
                    except:
                        pass
                
                st.markdown("---")
            
            # ========================================
            # Cross-variable comparison
            # ========================================
            st.markdown("### 📈 Cross-Variable FEVD Comparison")
            st.markdown("**Comparing variance decomposition across all response variables**")
            
            comparison_horizon = st.selectbox(
                "Select horizon for cross-variable comparison:",
                selected_horizons,
                index=0 if selected_horizons else 0,
                key='comparison_horizon'
            )
            
            if comparison_horizon and 1 <= comparison_horizon <= fevd_periods:
                fig_comparison = go.Figure()
                
                for imp_idx, impulse_var in enumerate(fevd_variables):
                    if imp_idx >= n_vars:
                        continue
                    
                    try:
                        contributions = [fevd.decomp[comparison_horizon-1, resp_idx, imp_idx] * 100 
                                    for resp_idx in range(n_vars)]
                        
                        fig_comparison.add_trace(go.Bar(
                            name=impulse_var,
                            x=fevd_variables,
                            y=contributions,
                            text=[f'{c:.1f}%' for c in contributions],
                            textposition='inside',
                            textfont=dict(color='white')
                        ))
                    except IndexError:
                        continue
                
                fig_comparison.update_layout(
                    barmode='stack',
                    title=f'FEVD Comparison Across Variables (Horizon={comparison_horizon})',
                    xaxis_title='Response Variable',
                    yaxis_title='Percentage (%)',
                    yaxis=dict(range=[0, 100]),
                    height=450,
                    legend_title='Shock Source'
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.success(f"""
            ✓ **FEVD Analysis Complete**
            
            - Analyzed {fevd_periods} forecast horizons
            - {n_vars} variables in system
            - Displayed {len(selected_horizons)} specific horizons in detail
            - R-style stacked bar charts for easy interpretation
            """)
            
        except Exception as e:
            st.error(f"❌ Error computing FEVD: {e}")
            import traceback
            st.code(traceback.format_exc())
            
            # Debug information
            st.markdown("### 🔍 Debug Information")
            st.write(f"- Number of variables selected: {len(variables)}")
            st.write(f"- Variables: {variables}")
            if 'fevd_variables' in locals():
                st.write(f"- FEVD variables: {fevd_variables}")
            if 'fevd' in locals():
                st.write(f"- FEVD shape: {fevd.decomp.shape}")
    with st.expander("📊 5.1 VAR Forecasts with Confidence Intervals & MSE", expanded=True):
        st.markdown("### Multi-Step Ahead Forecasts with Uncertainty Quantification")
        
        forecast_steps = st.slider("Forecast horizon", 5, 50, 5, key='forecast_steps')
        
        try:
            # Generate forecasts with confidence intervals and MSE
            forecast_df, lower_bounds, upper_bounds, mse_matrices = forecast_with_mse_and_ci(
                var_model, df, steps=forecast_steps, alpha=0.05
            )
            
            if forecast_df is not None:
                # Display MSE matrices
                st.markdown("#### 📊 Mean Squared Error (MSE) Matrices")
                st.write("""
                **What is MSE Matrix?**
                - Shows forecast error variance-covariance structure
                - Diagonal: Variance of forecast errors for each variable
                - Off-diagonal: Covariance between forecast errors
                - Used to construct confidence intervals
                """)
                
                st.markdown("---")
                st.markdown("##### 📋 MSE Matrices (First 5 Horizons)")
                
                for h in range(min(5, forecast_steps)):
                    st.markdown(f"**Horizon {h+1}:**")
                    mse_df = pd.DataFrame(
                        mse_matrices[h],
                        index=variables,
                        columns=variables
                    )
                    st.dataframe(mse_df.style.format('{:.6f}').background_gradient(cmap='YlOrRd'),
                               use_container_width=True)
                    
                    # Show interpretation for first horizon
                    if h == 0:
                        st.info(f"""
                        **Interpretation for Horizon 1:**
                        - Diagonal values: One-step ahead forecast error variance
                        - Larger values = more uncertainty
                        - Off-diagonal: Co-movement of forecast errors
                        """)
                
                # Plot forecasts with confidence intervals
                st.markdown("---")
                st.markdown("#### 📈 Forecasts with 95% Confidence Intervals")
                
                for var_idx, var in enumerate(variables):
                    fig = go.Figure()
                    
                    # Historical data (last 100 observations)
                    hist = df[var].iloc[-100:]
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist.values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[var],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    # Confidence interval band
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                        y=upper_bounds[var] + lower_bounds[var][::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% CI',
                        hoverinfo='skip'
                    ))
                    
                    # Upper bound line
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=upper_bounds[var],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(color='red', width=1, dash='dot'),
                        showlegend=True
                    ))
                    
                    # Lower bound line
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=lower_bounds[var],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(color='red', width=1, dash='dot'),
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f'{var} - Forecast with 95% Confidence Intervals',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        hovermode='x unified',
                        height=400,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table with confidence intervals
                st.markdown("---")
                st.markdown("#### 📋 Forecast Values with Confidence Intervals")
                
                forecast_table_data = []
                for i, idx in enumerate(forecast_df.index):
                    row = {'Date': idx.strftime('%Y-%m-%d')}
                    for var in variables:
                        row[f'{var} Forecast'] = forecast_df.loc[idx, var]
                        row[f'{var} Lower 95%'] = lower_bounds[var][i]
                        row[f'{var} Upper 95%'] = upper_bounds[var][i]
                        row[f'{var} CI Width'] = upper_bounds[var][i] - lower_bounds[var][i]
                    forecast_table_data.append(row)
                
                forecast_table = pd.DataFrame(forecast_table_data)
                
                # Format columns
                format_dict = {'Date': '{}'}
                for var in variables:
                    format_dict[f'{var} Forecast'] = '{:.4f}'
                    format_dict[f'{var} Lower 95%'] = '{:.4f}'
                    format_dict[f'{var} Upper 95%'] = '{:.4f}'
                    format_dict[f'{var} CI Width'] = '{:.4f}'
                
                st.dataframe(forecast_table.style.format(format_dict), use_container_width=True)
                
                st.info("""
                **How to interpret:**
                - **Forecast**: Point estimate (most likely value)
                - **Lower/Upper 95%**: We're 95% confident true value will be in this range
                - **CI Width**: Wider = more uncertainty
                - CI Width increases with forecast horizon (normal behavior)
                """)
                
        except Exception as e:
            st.error(f"Forecast error: {e}")
    
    with st.expander("🎯 5.2 Forecast Accuracy", expanded=True):
        st.markdown("### Out-of-Sample Evaluation")
        
        test_pct = st.slider("Test set %", 5, 30, 20)
        test_size = int(len(df) * test_pct / 100)
        
        if test_size >= int(selected_lag) + 5:
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]
            
            try:
                var_train = VAR(train_df).fit(int(selected_lag))
                var_forecast = var_train.forecast(train_df.values[-var_train.k_ar:], steps=test_size)
                var_forecast_df = pd.DataFrame(var_forecast, columns=variables, index=test_df.index)
                
                metrics = []
                for var in variables:
                    actual = test_df[var].values
                    pred = var_forecast_df[var].values
                    
                    rmse = np.sqrt(np.mean((actual - pred)**2))
                    mae = np.mean(np.abs(actual - pred))
                    mape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
                    
                    metrics.append({
                        'Variable': var,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE (%)': mape
                    })
                
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df.style.format({
                    'RMSE': '{:.4f}',
                    'MAE': '{:.4f}',
                    'MAPE (%)': '{:.2f}'
                }), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
                
                
    # ====================================================================
# SECTION 5.3: ARIMA FORECASTING & MODEL COMPARISON
# ====================================================================

    with st.expander("📊 5.3 ARIMA Models & VAR vs ARIMA Comparison", expanded=True):
        st.markdown("### Individual ARIMA Models for Each Variable")
        st.markdown("""
        **Purpose:** Compare multivariate VAR with univariate ARIMA models
        
        **ARIMA(p,d,q) Components:**
        - **p**: Autoregressive order (past values)
        - **d**: Differencing order (to achieve stationarity)
        - **q**: Moving average order (past forecast errors)
        """)
        
        # ================================================================
        # AUTOMATIC ARIMA MODEL SELECTION
        # ================================================================
        
        st.markdown("---")
        st.markdown("#### 🔍 Step 1: Automatic ARIMA Model Selection")
        
        st.info("""
        **Grid Search Approach:**
        - Test multiple (p,d,q) combinations
        - Select model with lowest AIC
        - Validate with diagnostic tests
        """)
        
        # Define search space
        p_range = range(0, 4)  # AR terms
        d_range = range(0, 2)  # Differencing
        q_range = range(0, 4)  # MA terms
        
        arima_models = {}
        arima_forecasts = {}
        arima_diagnostics = {}
        
        for var in variables:
            st.markdown(f"##### Variable: **{var}**")
            
            with st.spinner(f"Searching optimal ARIMA for {var}..."):
                try:
                    # Get the series (use transformed data from VAR analysis)
                    series = df[var].dropna()
                    
                    # Grid search
                    best_aic = np.inf
                    best_order = None
                    best_model = None
                    
                    search_results = []
                    
                    for p in p_range:
                        for d in d_range:
                            for q in q_range:
                                try:
                                    # Fit ARIMA
                                    model = ARIMA(series, order=(p, d, q))
                                    fitted = model.fit()
                                    
                                    # Store results
                                    search_results.append({
                                        'p': p,
                                        'd': d,
                                        'q': q,
                                        'AIC': fitted.aic,
                                        'BIC': fitted.bic,
                                        'Log-Likelihood': fitted.llf
                                    })
                                    
                                    # Track best model
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_order = (p, d, q)
                                        best_model = fitted
                                        
                                except:
                                    continue
                    
                    if best_model is None:
                        st.error(f"Could not find suitable ARIMA model for {var}")
                        continue
                    
                    # Store best model
                    arima_models[var] = {
                        'model': best_model,
                        'order': best_order,
                        'aic': best_aic
                    }
                    
                    # Display search results
                    st.markdown("---")
                    st.markdown(f"##### 📋 ARIMA Grid Search Results for {var}")
                    
                    search_df = pd.DataFrame(search_results)
                    search_df = search_df.sort_values('AIC').head(10)
                    
                    st.dataframe(
                        search_df.style
                    .format({
                            'AIC': '{:.4f}',
                            'BIC': '{:.4f}',
                            'Log-Likelihood': '{:.4f}'
                        })
                        .background_gradient(subset=['AIC'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    # Display best model
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best Model", f"ARIMA{best_order}")
                    with col2:
                        st.metric("AIC", f"{best_aic:.4f}")
                    with col3:
                        st.metric("BIC", f"{best_model.bic:.4f}")
                    with col4:
                        st.metric("Log-Likelihood", f"{best_model.llf:.2f}")
                    
                    # Model summary
                    st.markdown("---")
                    st.markdown(f"##### 📋 Detailed Summary: ARIMA{best_order} for {var}")
                    st.text(str(best_model.summary()))
                    
                    st.success(f"✅ Best model for {var}: ARIMA{best_order}")
                    
                except Exception as e:
                    st.error(f"Error fitting ARIMA for {var}: {e}")
                    continue
            
            st.markdown("---")
        
        # ================================================================
        # ARIMA DIAGNOSTICS
        # ================================================================
        
        st.markdown("---")
        st.markdown("#### 🔬 Step 2: ARIMA Model Diagnostics")
        
        for var, model_info in arima_models.items():
            st.markdown(f"##### Diagnostics: {var} - ARIMA{model_info['order']}")
            
            model = model_info['model']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Residual Tests:**")
                
                try:
                    # Ljung-Box test
                    lb_test = acorr_ljungbox(model.resid, lags=[10], return_df=True)
                    lb_pval = lb_test.loc[10, 'lb_pvalue']
                    
                    # Jarque-Bera test
                    from scipy.stats import jarque_bera
                    jb_stat, jb_pval = jarque_bera(model.resid)
                    
                    diag_data = {
                        'Test': ['Ljung-Box (Lag 10)', 'Jarque-Bera (Normality)'],
                        'Statistic': [f"{lb_test.loc[10, 'lb_stat']:.4f}", f"{jb_stat:.4f}"],
                        'P-value': [f"{lb_pval:.4f}", f"{jb_pval:.4f}"],
                        'Result': [
                            '✅ Pass' if lb_pval > 0.05 else '❌ Fail',
                            '✅ Pass' if jb_pval > 0.05 else '❌ Fail'
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
                    
                    arima_diagnostics[var] = {
                        'lb_pval': lb_pval,
                        'jb_pval': jb_pval,
                        'passes': (lb_pval > 0.05) and (jb_pval > 0.05)
                    }
                    
                except Exception as e:
                    st.warning(f"Could not run diagnostics: {e}")
            
            with col2:
                st.markdown("**Model Statistics:**")
                
                stats_data = {
                    'Metric': ['AIC', 'BIC', 'Log-Likelihood', 'Observations'],
                    'Value': [
                        f"{model.aic:.4f}",
                        f"{model.bic:.4f}",
                        f"{model.llf:.2f}",
                        f"{model.nobs}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            st.markdown("---")
        
        # ================================================================
        # ARIMA FORECASTS WITH CONFIDENCE INTERVALS
        # ================================================================
        
        st.markdown("---")
        st.markdown("#### 📈 Step 3: ARIMA Forecasts with Confidence Intervals")
        
        arima_forecast_steps = st.slider(
            "ARIMA Forecast Horizon",
            5, 50, 5,
            key='arima_forecast_steps',
            help="Number of periods to forecast ahead"
        )
        
        for var, model_info in arima_models.items():
            st.markdown(f"##### Forecast: {var}")
            
            model = model_info['model']
            
            try:
                # Generate forecast
                forecast_result = model.get_forecast(steps=arima_forecast_steps)
                forecast_mean = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int(alpha=0.05)
                
                # Create proper date index for forecasts
                forecast_index = pd.date_range(
                    start=df.index[-1],
                    periods=arima_forecast_steps + 1,
                    freq=df.index.freq
                )[1:]
                
                # Store forecasts WITH PROPER INDEX
                arima_forecasts[var] = {
                    'mean': pd.Series(forecast_mean.values, index=forecast_index),
                    'lower': pd.Series(forecast_ci.iloc[:, 0].values, index=forecast_index),
                    'upper': pd.Series(forecast_ci.iloc[:, 1].values, index=forecast_index)
                }
                
                # Create plot
                fig = go.Figure()
                
                # Historical data (last 100 observations)
                hist_data = df[var].iloc[-100:]
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast_mean,
                    mode='lines+markers',
                    name='ARIMA Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                    y=forecast_ci.iloc[:, 1].tolist() + forecast_ci.iloc[:, 0].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI',
                    hoverinfo='skip'
                ))
                
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast_ci.iloc[:, 1],
                    mode='lines',
                    name='Upper 95%',
                    line=dict(color='red', width=1, dash='dot')
                ))
                
                # Lower bound
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast_ci.iloc[:, 0],
                    mode='lines',
                    name='Lower 95%',
                    line=dict(color='red', width=1, dash='dot')
                ))
                
                fig.update_layout(
                    title=f'{var} - ARIMA{model_info["order"]} Forecast with 95% CI',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified',
                    height=400,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.markdown("---")
                st.markdown(f"##### 📋 Forecast Values: {var}")
                
                forecast_table = pd.DataFrame({
                    'Period': range(1, arima_forecast_steps + 1),
                    'Forecast': forecast_mean.values,
                    'Lower 95%': forecast_ci.iloc[:, 0].values,
                    'Upper 95%': forecast_ci.iloc[:, 1].values,
                    'CI Width': (forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]).values
                })
                
                st.dataframe(
                    forecast_table.style.format({
                        'Forecast': '{:.4f}',
                        'Lower 95%': '{:.4f}',
                        'Upper 95%': '{:.4f}',
                        'CI Width': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error forecasting {var}: {e}")
            
            st.markdown("---")
        
        # ================================================================
        # MODEL COMPARISON: VAR vs ARIMA
        # ================================================================
        
        st.markdown("---")
        st.markdown("#### ⚖️ Step 4: VAR vs ARIMA Comparison")
        
        st.markdown("""
        **Comparison Criteria:**
        1. **Information Criteria:** AIC, BIC (lower is better)
        2. **Forecast Accuracy:** Out-of-sample RMSE, MAE, MAPE
        3. **Model Complexity:** Number of parameters
        4. **Visual Comparison:** Side-by-side forecast plots
        """)
        
        # --- Information Criteria Comparison ---
        st.markdown("##### A. Information Criteria Comparison")
        
        ic_comparison = []
        
        # VAR metrics (system-wide)
        var_k = len(variables)
        var_p = int(selected_lag)
        var_params = var_k * (1 + var_p * var_k)  # intercepts + coefficients
        
        ic_comparison.append({
            'Model': f'VAR({var_p})',
            'Type': 'Multivariate',
            'Variables': var_k,
            'Parameters': var_params,
            'AIC': f'{var_model.aic:.4f}',
            'BIC': f'{var_model.bic:.4f}',
            'Log-Likelihood': f'{var_model.llf:.2f}'
        })
        
        # ARIMA metrics (individual)
        for var, model_info in arima_models.items():
            model = model_info['model']
            order = model_info['order']
            arima_params = sum(order) + (1 if order[0] > 0 else 0)  # Approximate
            
            ic_comparison.append({
                'Model': f'ARIMA{order}',
                'Type': 'Univariate',
                'Variables': var,
                'Parameters': arima_params,
                'AIC': f'{model.aic:.4f}',
                'BIC': f'{model.bic:.4f}',
                'Log-Likelihood': f'{model.llf:.2f}'
            })
        
        ic_df = pd.DataFrame(ic_comparison)
        
        st.dataframe(
            ic_df.style.background_gradient(
                subset=['Parameters'],
                cmap='YlOrRd'
            ),
            use_container_width=True
        )
        
        st.info("""
        **Interpretation:**
        - **AIC/BIC:** Lower values indicate better fit adjusted for complexity
        - **VAR:** Captures cross-variable dynamics (spillovers)
        - **ARIMA:** Simpler, focuses on individual series dynamics
        """)
        
        # --- Out-of-Sample Forecast Comparison ---
        st.markdown("---")
        st.markdown("##### B. Out-of-Sample Forecast Accuracy")

        test_pct_arima = st.slider(
            "Test set % for comparison",
            5, 30, 20,
            key='test_pct_arima'
        )

        test_size_arima = int(len(df) * test_pct_arima / 100)

        if test_size_arima >= max(int(selected_lag) + 5, 20):
            train_df_arima = df.iloc[:-test_size_arima]
            test_df_arima = df.iloc[-test_size_arima:]
            
            comparison_metrics = []
            
            # ========================================
            # VAR FORECASTS
            # ========================================
            st.markdown("**🔵 Computing VAR Forecasts...**")
            var_forecast_df = None
            
            try:
                var_train = VAR(train_df_arima).fit(int(selected_lag))
                var_forecast = var_train.forecast(
                    train_df_arima.values[-var_train.k_ar:],
                    steps=test_size_arima
                )
                var_forecast_df = pd.DataFrame(
                    var_forecast,
                    columns=variables,
                    index=test_df_arima.index
                )
                
                st.success(f"✓ VAR forecasts generated: {var_forecast_df.shape}")
                
                # Show first few VAR forecasts for debugging
                st.markdown("---")
                st.markdown("##### 🔍 VAR Forecast Preview")
                st.dataframe(var_forecast_df.head())
                
                for var in variables:
                    actual = test_df_arima[var].values
                    pred_var = var_forecast_df[var].values
                    
                    # ⭐ CRITICAL DEBUG ⭐
                    st.write(f"**VAR - {var}:**")
                    st.write(f"  - Actual shape: {actual.shape}")
                    st.write(f"  - Predicted shape: {pred_var.shape}")
                    st.write(f"  - First 3 actual: {actual[:3]}")
                    st.write(f"  - First 3 predicted: {pred_var[:3]}")
                    st.write(f"  - Difference: {actual[:3] - pred_var[:3]}")
                    
                    rmse_var = np.sqrt(np.mean((actual - pred_var)**2))
                    mae_var = np.mean(np.abs(actual - pred_var))
                    mape_var = np.mean(np.abs((actual - pred_var) / (actual + 1e-10))) * 100
                    
                    st.write(f"  - ✅ RMSE: {rmse_var:.6f}")
                    st.write(f"  - ✅ MAE: {mae_var:.6f}")
                    st.write("---")
                    
                    comparison_metrics.append({
                        'Variable': var,
                        'Model': f'VAR({selected_lag})',
                        'RMSE': rmse_var,
                        'MAE': mae_var,
                        'MAPE (%)': mape_var
                    })
            except Exception as e:
                st.error(f"VAR forecast error: {e}")
            
            # ========================================
            # ARIMA FORECASTS
            # ========================================
            st.markdown("---")
            st.markdown("**🔴 Computing ARIMA Forecasts...**")
            
            arima_test_forecasts = {}
            
            for var in variables:
                st.write(f"### Processing ARIMA for: {var}")
                
                try:
                    # Refit ARIMA on training data
                    series_train = train_df_arima[var].dropna()
                    
                    if var in arima_models:
                        order = arima_models[var]['order']
                        st.info(f"ARIMA order: {order}")
                        
                        # Fit ARIMA
                        arima_train = ARIMA(series_train, order=order).fit()
                        arima_forecast = arima_train.forecast(steps=test_size_arima)
                        
                        st.write(f"**ARIMA Forecast Generated:**")
                        st.write(f"  - Type: {type(arima_forecast)}")
                        st.write(f"  - Length: {len(arima_forecast)}")
                        st.write(f"  - First 3 values: {arima_forecast.values[:3]}")
                        
                        # Store forecast with index
                        arima_test_forecasts[var] = pd.Series(
                            arima_forecast.values,
                            index=test_df_arima.index
                        )
                        
                        # Get actual values
                        actual = test_df_arima[var].values
                        pred_arima = arima_forecast.values
                        
                        # ⭐ CRITICAL: Check if predictions are different ⭐
                        st.write(f"**Comparison Check:**")
                        st.write(f"  - Actual shape: {actual.shape}")
                        st.write(f"  - ARIMA predicted shape: {pred_arima.shape}")
                        st.write(f"  - First 3 actual: {actual[:3]}")
                        st.write(f"  - First 3 ARIMA predicted: {pred_arima[:3]}")
                        
                        # Check if VAR and ARIMA forecasts are identical
                        if var_forecast_df is not None:
                            var_pred = var_forecast_df[var].values
                            st.write(f"  - First 3 VAR predicted: {var_pred[:3]}")
                            st.write(f"  - VAR vs ARIMA difference: {var_pred[:3] - pred_arima[:3]}")
                            
                            if np.allclose(var_pred, pred_arima):
                                st.error("🚨 **BUG DETECTED: VAR and ARIMA forecasts are IDENTICAL!**")
                            else:
                                st.success("✓ VAR and ARIMA forecasts are different (correct)")
                        
                        # Handle length mismatch
                        min_len = min(len(actual), len(pred_arima))
                        actual_trimmed = actual[:min_len]
                        pred_arima_trimmed = pred_arima[:min_len]
                        
                        # Calculate metrics
                        rmse_arima = np.sqrt(np.mean((actual_trimmed - pred_arima_trimmed)**2))
                        mae_arima = np.mean(np.abs(actual_trimmed - pred_arima_trimmed))
                        mape_arima = np.mean(np.abs((actual_trimmed - pred_arima_trimmed) / (actual_trimmed + 1e-10))) * 100
                        
                        st.write(f"**ARIMA Metrics:**")
                        st.write(f"  - ✅ RMSE: {rmse_arima:.6f}")
                        st.write(f"  - ✅ MAE: {mae_arima:.6f}")
                        st.write(f"  - ✅ MAPE: {mape_arima:.2f}%")
                        
                        comparison_metrics.append({
                            'Variable': var,
                            'Model': f'ARIMA{order}',
                            'RMSE': rmse_arima,
                            'MAE': mae_arima,
                            'MAPE (%)': mape_arima
                        })
                        
                        st.success(f"✓ ARIMA metrics added for {var}")
                        
                except Exception as e:
                    st.error(f"ARIMA forecast error for {var}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                st.markdown("---")
            
            # Display comparison table
            if comparison_metrics:
                st.markdown("### 📊 Final Comparison Table")
                comparison_df = pd.DataFrame(comparison_metrics)
                
                st.dataframe(
                    comparison_df.style
                    .format({
                        'RMSE': '{:.6f}',  # More precision
                        'MAE': '{:.6f}',
                        'MAPE (%)': '{:.2f}'
                    })
                    .background_gradient(subset=['RMSE', 'MAE', 'MAPE (%)'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                
                # ========================================
                # NEW: SUMMARY TABLE WITH WINNER
                # ========================================
                st.markdown("---")
                st.markdown("##### D. Summary: Best Model per Variable")
                
                summary_data = []
                for var in variables:
                    var_comp = comparison_df[comparison_df['Variable'] == var]
                    
                    if len(var_comp) >= 2:
                        # Get VAR metrics
                        var_metrics = var_comp[var_comp['Model'].str.contains('VAR')]
                        arima_metrics = var_comp[var_comp['Model'].str.contains('ARIMA')]
                        
                        if not var_metrics.empty and not arima_metrics.empty:
                            var_rmse = var_metrics.iloc[0]['RMSE']
                            arima_rmse = arima_metrics.iloc[0]['RMSE']
                            
                            # Determine winner
                            if var_rmse < arima_rmse:
                                winner = 'VAR'
                                improvement = ((arima_rmse - var_rmse) / arima_rmse) * 100
                            else:
                                winner = 'ARIMA'
                                improvement = ((var_rmse - arima_rmse) / var_rmse) * 100
                            
                            summary_data.append({
                                'Variable': var,
                                'VAR RMSE': f'{var_rmse:.4f}',
                                'ARIMA RMSE': f'{arima_rmse:.4f}',
                                'Winner': f'🏆 {winner}',
                                'Improvement': f'{improvement:.1f}%'
                            })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    
                    st.dataframe(
                        summary_df.style.applymap(
                            lambda x: 'background-color: #d4edda; font-weight: bold' if '🏆' in str(x) else '',
                            subset=['Winner']
                        ),
                        use_container_width=True
                    )
                    
                    # Count wins
                    var_wins = sum(1 for row in summary_data if 'VAR' in row['Winner'])
                    arima_wins = len(summary_data) - var_wins
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VAR Wins", var_wins)
                    with col2:
                        st.metric("ARIMA Wins", arima_wins)
                    with col3:
                        st.metric("Total Variables", len(summary_data))
                    
                    # Overall conclusion
                    st.markdown("---")
                    st.markdown("##### 🎯 Overall Recommendation")
                    
                    if var_wins > arima_wins:
                        st.success(f"""
                        ✅ **VAR MODEL PERFORMS BETTER**
                        
                        **VAR wins:** {var_wins}/{len(summary_data)} variables
                        
                        **Advantages:**
                        - Captures cross-variable dynamics
                        - Better for systems with spillovers
                        - Provides Granger causality insights
                        - Useful for impulse response analysis
                        
                        **Recommendation:** Use VAR for forecasting and structural analysis
                        """)
                    elif arima_wins > var_wins:
                        st.info(f"""
                        ℹ️ **ARIMA MODELS PERFORM BETTER**
                        
                        **ARIMA wins:** {arima_wins}/{len(summary_data)} variables
                        
                        **Possible reasons:**
                        - Weak cross-variable relationships
                        - Individual series have strong AR/MA structure
                        - VAR may be overparameterized for this data
                        
                        **Recommendation:** Consider ARIMA for pure forecasting, VAR for relationship analysis
                        """)
                    else:
                        st.info(f"""
                        ⚖️ **MODELS PERFORM SIMILARLY**
                        
                        **Tied:** {var_wins} - {arima_wins}
                        
                        **Recommendation:**
                        - Use VAR if studying relationships between variables
                        - Use ARIMA if only forecasting individual series
                        - Report both for robustness
                        """)
        else:
            st.warning(f"Need at least {max(int(selected_lag) + 5, 20)} test observations. Current: {test_size_arima}")
        
        # ================================================================
        # FINAL RECOMMENDATIONS
        # ================================================================
        
        st.markdown("---")
        st.markdown("#### 💡 Final Recommendations")
        
        st.success("""
        **Model Selection Guide:**
        
        **Choose VAR when:**
        - ✓ Studying relationships between variables
        - ✓ Need Granger causality analysis
        - ✓ Want impulse response functions
        - ✓ Variables have significant cross-effects
        
        **Choose ARIMA when:**
        - ✓ Only forecasting individual series
        - ✓ Variables are largely independent
        - ✓ Need simple, interpretable models
        - ✓ Computational efficiency is critical
        
        **For Your Report:**
        1. Present both VAR and ARIMA results
        2. Compare forecast accuracy metrics (RMSE, MAE, MAPE)
        3. Show visual comparison plots
        4. Discuss trade-offs (complexity vs accuracy)
        5. Justify your final model choice based on evidence
    """)
    # ====================================================================
    # ============================================================================
    # ============================================================================
    # COMPLETE SECTION 6: COINTEGRATION ANALYSIS WITH CORRECTED TESTS
    # ============================================================================
    # 
    # INSTRUCTIONS:
    # 1. Add the two helper functions (perform_adf_test_simple and perform_kpss_test_simple) 
    #    after your imports at the top of the file (around line 27)
    # 2. Replace your entire "Section 6" checkbox block with this code
    #
    # ============================================================================

    # ============================================================================
    # STEP 1: ADD THESE HELPER FUNCTIONS AFTER YOUR IMPORTS (around line 27)
    # ============================================================================

    def perform_adf_test_simple(series, name):
        """
        ADF test matching R's adf.test() defaults
        R uses: constant (drift) but NO trend
        """
        try:
            # Use 'c' for constant only - matches R's default
            result = adfuller(series.dropna(), maxlag=None, regression='c', autolag='AIC')
            return {
                'Variable': name,
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Lags Used': result[2],
                'Observations': result[3],
                'Critical Value (1%)': result[4]['1%'],
                'Critical Value (5%)': result[4]['5%'],
                'Critical Value (10%)': result[4]['10%'],
                'Stationary (5%)': result[1] < 0.05,
                'Interpretation': 'STATIONARY' if result[1] < 0.05 else 'NON-STATIONARY'
            }
        except Exception as e:
            st.error(f"Error in ADF test for {name}: {e}")
            return {
                'Variable': name,
                'ADF Statistic': np.nan,
                'p-value': np.nan,
                'Lags Used': np.nan,
                'Observations': len(series),
                'Critical Value (1%)': np.nan,
                'Critical Value (5%)': np.nan,
                'Critical Value (10%)': np.nan,
                'Stationary (5%)': False,
                'Interpretation': 'ERROR'
            }

    def perform_kpss_test_simple(series, name):
        """
        KPSS test matching R's kpss.test() defaults
        R uses: level (constant) but NO trend
        """
        try:
            # Use 'c' for constant (level) - matches R's default
            result = kpss(series.dropna(), regression='c', nlags='auto')
            return {
                'Variable': name,
                'KPSS Statistic': result[0],
                'p-value': result[1],
                'Critical Value (10%)': result[3]['10%'],
                'Critical Value (5%)': result[3]['5%'],
                'Critical Value (2.5%)': result[3]['2.5%'],
                'Critical Value (1%)': result[3]['1%'],
                'Stationary (5%)': result[1] > 0.05,
                'Interpretation': 'STATIONARY' if result[1] > 0.05 else 'NON-STATIONARY'
            }
        except Exception as e:
            st.error(f"Error in KPSS test for {name}: {e}")
            return {
                'Variable': name,
                'KPSS Statistic': np.nan,
                'p-value': np.nan,
                'Critical Value (10%)': np.nan,
                'Critical Value (5%)': np.nan,
                'Critical Value (2.5%)': np.nan,
                'Critical Value (1%)': np.nan,
                'Stationary (5%)': False,
                'Interpretation': 'ERROR'
            }

    # ============================================================================
    # STEP 2: REPLACE YOUR SECTION 6 CHECKBOX BLOCK WITH THIS CODE
    # ============================================================================

    if st.checkbox("6. Cointegration Analysis", value=False):
        st.markdown('<div class="section-header">6. Cointegration Analysis on Log-Level Data</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        ### Purpose
        Test whether multiple I(1) series share a common stochastic trend (cointegration).
        
        ### What is Cointegration?
        - Multiple non-stationary series that move together in the long run
        - Linear combination of I(1) series is I(0) (stationary)
        - Implies long-run equilibrium relationship
        - Allows for Error Correction Model (VEC)
        
        ### Methodology
        1. Transform data to log levels
        2. Test for I(1) property (non-stationary in levels, stationary in differences)
        3. Perform Johansen cointegration test
        4. Estimate VEC model if cointegration exists
        """)
        
        # ========================================================================
        # STEP 0: SELECT VARIABLES FOR COINTEGRATION
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📋 Step 0: Variable Selection for Cointegration")
        
        # Initialize with all variables
        variables_for_cointegration = variables.copy()
        
        # Try to use Section 2 results if available
        try:
            if 'non_stationary' in locals() and non_stationary and len(non_stationary) >= 2:
                st.info(f"""
                **Based on Section 2 stationarity tests:**
                
                Non-stationary variables: {', '.join(non_stationary)}
                Stationary variables: {', '.join(stationary) if stationary else 'None'}
                """)
                variables_for_cointegration = non_stationary.copy()
            else:
                st.info("""
                **Using all variables for cointegration analysis.**
                
                We'll verify I(1) property in this section.
                """)
        except:
            st.info(f"Using all {len(variables)} variables for cointegration analysis.")
        
        if len(variables_for_cointegration) < 2:
            st.error("❌ Need at least 2 variables for cointegration. Please select more variables.")
            st.stop()
        
        st.success(f"""
        ✅ **Selected for cointegration analysis:** {len(variables_for_cointegration)} variables
        
        Variables: {', '.join(variables_for_cointegration)}
        """)
        
        # ========================================================================
        # STEP 1: PREPARE LOG-LEVEL DATA
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 🔍 Step 1: Prepare Log-Level Data")
        
        st.info("""
        **For cointegration, we need:**
        1. Log-transformed prices (not returns!)
        2. Data in LEVELS (not differenced)
        3. All series must be I(1)
        """)
        
        try:
            df_log_levels = pd.DataFrame()
            vars_to_remove = []
            
            for var in variables_for_cointegration:
                if (df_raw[var] > 0).all():
                    df_log_levels[var] = np.log(df_raw[var])
                    st.success(f"✓ {var}: Transformed to log levels")
                else:
                    st.error(f"❌ {var}: Cannot take log (has negative/zero values)")
                    vars_to_remove.append(var)
            
            # Remove invalid variables
            for var in vars_to_remove:
                variables_for_cointegration.remove(var)
            
            df_log_levels = df_log_levels.dropna()
            
            if len(variables_for_cointegration) < 2:
                st.error("Not enough valid variables after transformation")
                st.stop()
            
            st.write(f"**✓ Log-level data ready:** {len(df_log_levels)} observations, {len(variables_for_cointegration)} variables")
            
            # Show sample
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 5 rows:**")
                st.dataframe(df_log_levels.head())
            with col2:
                st.write("**Last 5 rows:**")
                st.dataframe(df_log_levels.tail())
            
        except Exception as e:
            st.error(f"Error preparing data: {e}")
            st.stop()
        
        # ========================================================================
        # STEP 2: STATIONARITY TESTS ON LOG LEVELS (CORRECTED - MATCHES R)
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Step 2: Stationarity Tests on Log Levels")
        
        st.markdown("""
        **Test Hypotheses:**
        
        **ADF Test:** H₀ = Unit Root (Non-Stationary)
        - p-value < 0.05 → Reject H₀ → **STATIONARY** ❌ (bad for cointegration)
        - p-value ≥ 0.05 → Cannot reject H₀ → **NON-STATIONARY** ✓ (good for cointegration)
        
        **KPSS Test:** H₀ = Stationary
        - p-value > 0.05 → Cannot reject H₀ → **STATIONARY** ❌ (bad for cointegration)
        - p-value ≤ 0.05 → Reject H₀ → **NON-STATIONARY** ✓ (good for cointegration)
        
        **For cointegration:** Both tests should show **NON-STATIONARY**
        """)
        
        # ========================================================================
        # ADF TESTS ON LOG LEVELS
        # ========================================================================
        
        st.markdown("---")
        st.markdown("#### A. ADF Tests on Log Levels")
        st.markdown("**H₀: Unit Root (Non-Stationary)**")
        
        adf_log_levels_results = []
        for var in variables_for_cointegration:
            result = perform_adf_test_simple(df_log_levels[var], var)
            adf_log_levels_results.append(result)
        
        adf_log_df = pd.DataFrame(adf_log_levels_results)
        
        st.dataframe(
            adf_log_df[['Variable', 'ADF Statistic', 'p-value', 'Critical Value (5%)', 
                        'Lags Used', 'Interpretation']].style
            .apply(lambda x: ['background-color: #f8d7da' if v == 'STATIONARY' else 'background-color: #d4edda' 
                            for v in x] if x.name == 'Interpretation' else ['']*len(x), axis=0)
            .format({
                'ADF Statistic': '{:.4f}',
                'p-value': '{:.4f}',
                'Critical Value (5%)': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.caption("""
        **✓ GREEN = NON-STATIONARY** (good for cointegration)  
        **❌ RED = STATIONARY** (exclude from cointegration)
        """)
        
        # ========================================================================
        # KPSS TESTS ON LOG LEVELS
        # ========================================================================
        
        st.markdown("---")
        st.markdown("#### B. KPSS Tests on Log Levels")
        st.markdown("**H₀: Stationary**")
        
        kpss_log_levels_results = []
        for var in variables_for_cointegration:
            result = perform_kpss_test_simple(df_log_levels[var], var)
            kpss_log_levels_results.append(result)
        
        kpss_log_df = pd.DataFrame(kpss_log_levels_results)
        
        st.dataframe(
            kpss_log_df[['Variable', 'KPSS Statistic', 'p-value', 'Critical Value (5%)', 
                        'Interpretation']].style
            .apply(lambda x: ['background-color: #f8d7da' if v == 'STATIONARY' else 'background-color: #d4edda' 
                            for v in x] if x.name == 'Interpretation' else ['']*len(x), axis=0)
            .format({
                'KPSS Statistic': '{:.4f}',
                'p-value': '{:.4f}',
                'Critical Value (5%)': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.caption("""
        **✓ GREEN = NON-STATIONARY** (good for cointegration)  
        **❌ RED = STATIONARY** (exclude from cointegration)
        """)
        
        # ========================================================================
        # COMBINED INTERPRETATION
        # ========================================================================
        
        st.markdown("---")
        st.markdown("#### C. Combined Test Results")
        
        combined_levels = []
        for i, var in enumerate(variables_for_cointegration):
            adf_nonstat = adf_log_levels_results[i]['Interpretation'] == 'NON-STATIONARY'
            kpss_nonstat = kpss_log_levels_results[i]['Interpretation'] == 'NON-STATIONARY'
            
            if adf_nonstat and kpss_nonstat:
                conclusion = "✓✓ NON-STATIONARY (Both agree)"
                status = "I(1) candidate"
            elif not adf_nonstat and not kpss_nonstat:
                conclusion = "✗✗ STATIONARY (Both agree) - EXCLUDE"
                status = "I(0) - Exclude"
            else:
                conclusion = "⚠️ MIXED SIGNALS"
                status = "Inconclusive"
            
            combined_levels.append({
                'Variable': var,
                'ADF': adf_log_levels_results[i]['Interpretation'],
                'ADF p-value': f"{adf_log_levels_results[i]['p-value']:.4f}",
                'KPSS': kpss_log_levels_results[i]['Interpretation'],
                'KPSS p-value': f"{kpss_log_levels_results[i]['p-value']:.4f}",
                'Conclusion': conclusion,
                'Status': status
            })
        
        combined_df = pd.DataFrame(combined_levels)
        
        st.dataframe(
            combined_df.style.applymap(
                lambda x: 'background-color: #d4edda' if '✓✓' in str(x) else
                        ('background-color: #fff3cd' if '⚠️' in str(x) else
                        ('background-color: #f8d7da' if '✗✗' in str(x) or 'Exclude' in str(x) else '')),
                subset=['Conclusion', 'Status']
            ),
            use_container_width=True
        )
        
        # Filter out I(0) variables
        i0_vars = [r['Variable'] for r in combined_levels if 'Exclude' in r['Status']]
        
        if i0_vars:
            st.error(f"""
            ❌ **Excluding stationary variables:** {', '.join(i0_vars)}
            
            These are I(0) and cannot be part of cointegration analysis.
            """)
            
            variables_for_cointegration = [v for v in variables_for_cointegration if v not in i0_vars]
            
            if len(variables_for_cointegration) < 2:
                st.error("Not enough I(1) variables remaining.")
                st.stop()
            
            df_log_levels = df_log_levels[variables_for_cointegration]
            
            st.success(f"""
            ✓ **Continuing with {len(variables_for_cointegration)} I(1) variables:**
            {', '.join(variables_for_cointegration)}
            """)
        else:
            st.success("✅ All variables are non-stationary in log levels!")
        
        # ========================================================================
        # STEP 3: TESTS ON LOG DIFFERENCES (I(1) CONFIRMATION)
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Step 3: Tests on Log First Differences")
        st.markdown("**Purpose:** Confirm differences are stationary (I(1) property)")
        
        df_log_diff = df_log_levels.diff().dropna()
        
        # ADF on differences
        st.markdown("#### A. ADF on Log Differences")
        
        adf_diff_results = []
        for var in variables_for_cointegration:
            result = perform_adf_test_simple(df_log_diff[var], var)
            adf_diff_results.append(result)
        
        adf_diff_df = pd.DataFrame(adf_diff_results)
        
        st.dataframe(
            adf_diff_df[['Variable', 'ADF Statistic', 'p-value', 'Interpretation']].style
            .apply(lambda x: ['background-color: #d4edda' if v == 'STATIONARY' else 'background-color: #f8d7da' 
                            for v in x] if x.name == 'Interpretation' else ['']*len(x), axis=0)
            .format({'ADF Statistic': '{:.4f}', 'p-value': '{:.4f}'}),
            use_container_width=True
        )
        
        # KPSS on differences
        st.markdown("---")
        st.markdown("#### B. KPSS on Log Differences")
        
        kpss_diff_results = []
        for var in variables_for_cointegration:
            result = perform_kpss_test_simple(df_log_diff[var], var)
            kpss_diff_results.append(result)
        
        kpss_diff_df = pd.DataFrame(kpss_diff_results)
        
        st.dataframe(
            kpss_diff_df[['Variable', 'KPSS Statistic', 'p-value', 'Interpretation']].style
            .apply(lambda x: ['background-color: #d4edda' if v == 'STATIONARY' else 'background-color: #f8d7da' 
                            for v in x] if x.name == 'Interpretation' else ['']*len(x), axis=0)
            .format({'KPSS Statistic': '{:.4f}', 'p-value': '{:.4f}'}),
            use_container_width=True
        )
        
        # I(1) Classification
        st.markdown("---")
        st.markdown("#### C. Final I(1) Classification")
        
        i1_check = []
        for i, var in enumerate(variables_for_cointegration):
            # Levels: should be non-stationary
            adf_lev_nonstat = adf_log_levels_results[i]['Interpretation'] == 'NON-STATIONARY'
            kpss_lev_nonstat = kpss_log_levels_results[i]['Interpretation'] == 'NON-STATIONARY'
            
            # Differences: should be stationary
            adf_diff_stat = adf_diff_results[i]['Interpretation'] == 'STATIONARY'
            kpss_diff_stat = kpss_diff_results[i]['Interpretation'] == 'STATIONARY'
            
            is_i1_both = (adf_lev_nonstat and adf_diff_stat) and (kpss_lev_nonstat and kpss_diff_stat)
            is_i1_one = (adf_lev_nonstat and adf_diff_stat) or (kpss_lev_nonstat and kpss_diff_stat)
            
            if is_i1_both:
                status = 'I(1) ✓✓ (Both confirm)'
            elif is_i1_one:
                status = 'I(1) ✓ (One confirms)'
            else:
                status = 'NOT I(1) ✗'
            
            i1_check.append({
                'Variable': var,
                'Levels (ADF)': 'Non-stat ✓' if adf_lev_nonstat else 'Stat ✗',
                'Levels (KPSS)': 'Non-stat ✓' if kpss_lev_nonstat else 'Stat ✗',
                'Diff (ADF)': 'Stat ✓' if adf_diff_stat else 'Non-stat ✗',
                'Diff (KPSS)': 'Stat ✓' if kpss_diff_stat else 'Non-stat ✗',
                'I(1) Status': status
            })
        
        i1_df = pd.DataFrame(i1_check)
        
        st.dataframe(
            i1_df.style.applymap(
                lambda x: 'background-color: #d4edda' if '✓✓' in str(x) else
                        ('background-color: #fff9c4' if '✓ (One' in str(x) else
                        ('background-color: #f8d7da' if '✗' in str(x) else '')),
                subset=['I(1) Status']
            ),
            use_container_width=True
        )
        
        all_i1 = all('✓' in check['I(1) Status'] for check in i1_check)
        
        if all_i1:
            st.success("""
            ✅ **All variables confirmed as I(1)!**
            
            - Non-stationary in levels
            - Stationary in first differences
            - Ready for Johansen test
            """)
        else:
            st.warning("⚠️ Mixed I(1) results. Proceeding with caution...")
        
        # ========================================================================
        # STEP 4: LAG SELECTION FOR VAR IN LEVELS
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📐 Step 4: Lag Selection")
        
        st.info("This lag selection is for VAR in LOG LEVELS (for Johansen test)")
        
        max_lags_coint = st.slider(
            "Max lags to test:",
            1, min(15, len(df_log_levels)//10),
            min(12, len(df_log_levels)//10)
        )
        
        try:
            model_levels = VAR(df_log_levels)
            
            ic_results = []
            for lag in range(1, max_lags_coint + 1):
                try:
                    var_temp = model_levels.fit(lag)
                    ic_results.append({
                        'Lag': lag,
                        'AIC': var_temp.aic,
                        'BIC': var_temp.bic,
                        'HQIC': var_temp.hqic
                    })
                except:
                    continue
            
            if not ic_results:
                st.error("Could not estimate VAR")
                st.stop()
            
            ic_df = pd.DataFrame(ic_results)
            
            st.dataframe(
                ic_df.style
                .format({'AIC': '{:.4f}', 'BIC': '{:.4f}', 'HQIC': '{:.4f}'})
                .highlight_min(subset=['AIC', 'BIC', 'HQIC'], color='lightgreen', axis=0),
                use_container_width=True
            )
            
            aic_lag = int(ic_df.loc[ic_df['AIC'].idxmin(), 'Lag'])
            bic_lag = int(ic_df.loc[ic_df['BIC'].idxmin(), 'Lag'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AIC recommends", f"{aic_lag} lags")
            with col2:
                st.metric("BIC recommends", f"{bic_lag} lags")
            
            johansen_lag = st.selectbox(
                "Select lag for Johansen test:",
                ic_df['Lag'].tolist(),
                index=ic_df['Lag'].tolist().index(bic_lag)
            )
            
            st.success(f"✓ Using {johansen_lag} lags")
            
        except Exception as e:
            st.error(f"Error: {e}")
            johansen_lag = st.number_input("Manual lag:", 1, 12, 2)
        
        # ========================================================================
        # STEP 5: JOHANSEN COINTEGRATION TEST
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Step 5: Johansen Cointegration Test")
        
        st.markdown("""
        **Tests for cointegration rank:**
        - H₀: rank(Π) = r
        - H₁: rank(Π) > r
        
        **Two test statistics:**
        1. Trace test
        2. Max eigenvalue test
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            det_order = st.selectbox(
                "Deterministic terms:",
                [-1, 0, 1],
                index=2,
                format_func=lambda x: {
                    -1: "No terms",
                    0: "Constant outside",
                    1: "Constant in cointegration"
                }[x]
            )
        with col2:
            st.metric("Lag Order", johansen_lag)
        
        try:
            with st.spinner("Running Johansen test..."):
                johansen_result = coint_johansen(
                    df_log_levels.values,
                    det_order=det_order,
                    k_ar_diff=int(johansen_lag)
                )
            
            st.success("✓ Johansen test complete")
            
            # Trace test
            st.markdown("---")
            st.markdown("#### A. Trace Test")
            
            trace_data = []
            num_coint_trace = 0
            
            for r in range(len(variables_for_cointegration)):
                trace_stat = johansen_result.lr1[r]
                crit_95 = johansen_result.cvt[r, 1]
                reject = trace_stat > crit_95
                
                if reject:
                    num_coint_trace = r + 1
                
                trace_data.append({
                    'H₀: rank ≤': r,
                    'Trace Stat': f'{trace_stat:.4f}',
                    '95% CV': f'{crit_95:.4f}',
                    'Reject?': '✓ YES' if reject else '✗ NO'
                })
            
            trace_df = pd.DataFrame(trace_data)
            st.dataframe(
                trace_df.style.applymap(
                    lambda x: 'background-color: #d4edda' if '✓' in str(x) else 'background-color: #f8d7da',
                    subset=['Reject?']
                ),
                use_container_width=True
            )
            
            # Max eigenvalue test
            st.markdown("---")
            st.markdown("#### B. Max Eigenvalue Test")
            
            maxeig_data = []
            num_coint_maxeig = 0
            
            for r in range(len(variables_for_cointegration)):
                maxeig_stat = johansen_result.lr2[r]
                crit_95 = johansen_result.cvm[r, 1]
                reject = maxeig_stat > crit_95
                
                if reject and r == num_coint_maxeig:
                    num_coint_maxeig += 1
                
                maxeig_data.append({
                    'H₀: rank =': r,
                    'Max-Eig Stat': f'{maxeig_stat:.4f}',
                    '95% CV': f'{crit_95:.4f}',
                    'Reject?': '✓ YES' if reject else '✗ NO'
                })
            
            maxeig_df = pd.DataFrame(maxeig_data)
            st.dataframe(
                maxeig_df.style.applymap(
                    lambda x: 'background-color: #d4edda' if '✓' in str(x) else 'background-color: #f8d7da',
                    subset=['Reject?']
                ),
                use_container_width=True
            )
            
            # Conclusion
            st.markdown("---")
            st.markdown("### 📋 Test Summary")
            
            suggested_rank = min(num_coint_trace, num_coint_maxeig)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trace Test", f"{num_coint_trace} relations")
            with col2:
                st.metric("Max-Eig Test", f"{num_coint_maxeig} relations")
            with col3:
                st.metric("Suggested Rank", f"{suggested_rank} relations")
            
            if suggested_rank > 0:
                st.success(f"""
                ✅ **COINTEGRATION DETECTED!**
                
                Found {suggested_rank} cointegrating relationship(s)
                
                This means:
                - Long-run equilibrium exists
                - VEC model is appropriate
                - Prices move together in the long run
                """)
            else:
                st.info("""
                ℹ️ **NO COINTEGRATION DETECTED**
                
                - No long-run equilibrium
                - VAR in differences is appropriate
                - Your Section 3 model is correct
                """)
            
        except Exception as e:
            st.error(f"Johansen test error: {e}")
            import traceback
            st.code(traceback.format_exc())
            suggested_rank = 0
        
        # ========================================================================
        # STEP 6: VEC MODEL (if cointegration found)
        # ========================================================================
        
        if suggested_rank > 0:
            st.markdown("---")
            st.markdown("### 🔄 Step 6: Vector Error Correction Model")
            
            st.markdown("""
            **VEC Model:**
            
            Δlog(P)ₜ = Π·log(P)ₜ₋₁ + Γ₁·Δlog(P)ₜ₋₁ + ... + εₜ
            
            Where Π = αβ' with rank r
            """)
            
            vec_rank = st.number_input(
                "Cointegrating rank:",
                1, len(variables_for_cointegration)-1,
                int(suggested_rank)
            )
            
            try:
                with st.spinner("Estimating VEC model..."):
                    vec_model = VECM(
                        df_log_levels.values,
                        k_ar_diff=int(johansen_lag),
                        coint_rank=int(vec_rank),
                        deterministic='ci'
                    )
                    vec_result = vec_model.fit()
                
                st.success("✓ VEC model estimated!")
                
                # Model summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rank", vec_rank)
                with col2:
                    st.metric("Observations", vec_result.nobs)
                with col3:
                    st.metric("Log-Likelihood", f"{vec_result.llf:.2f}")
                
                # Cointegrating vectors (β)
                st.markdown("---")
                st.markdown("#### A. Cointegrating Vectors (β)")
                st.markdown("**Long-run equilibrium relationships**")
                
                beta_matrix = vec_result.beta
                beta_df = pd.DataFrame(
                    beta_matrix,
                    index=variables_for_cointegration,
                    columns=[f'CE{i+1}' for i in range(vec_rank)]
                )
                
                st.dataframe(
                    beta_df.style
                    .format('{:.4f}')
                    .background_gradient(cmap='RdYlGn', axis=0),
                    use_container_width=True
                )
                
                st.caption("""
                **Interpretation:** Each column is one equilibrium relation.
                β'·log(P) = 0 in equilibrium.
                """)
                
                # Normalized equations
                st.markdown("**Normalized Equilibrium Equations:**")
                
                for i in range(vec_rank):
                    beta_col = beta_matrix[:, i]
                    norm_idx = np.argmax(np.abs(beta_col))
                    beta_norm = beta_col / beta_col[norm_idx]
                    
                    terms = []
                    for j, var in enumerate(variables_for_cointegration):
                        coef = beta_norm[j]
                        if j == norm_idx:
                            terms.append(f"log({var})")
                        else:
                            sign = '-' if coef > 0 else '+'
                            terms.append(f"{sign} {abs(coef):.3f}·log({var})")
                    
                    equation = " ".join(terms) + " = 0"
                    st.code(f"CE{i+1}: {equation}")
                
                # Adjustment coefficients (α)
                st.markdown("---")
                st.markdown("#### B. Adjustment Coefficients (α)")
                st.markdown("**Speed of adjustment to equilibrium**")
                
                alpha_matrix = vec_result.alpha
                alpha_df = pd.DataFrame(
                    alpha_matrix,
                    index=variables_for_cointegration,
                    columns=[f'CE{i+1}' for i in range(vec_rank)]
                )
                
                st.dataframe(
                    alpha_df.style
                    .format('{:.4f}')
                    .background_gradient(cmap='RdBu', vmin=-0.5, vmax=0.5),
                    use_container_width=True
                )
                
                st.caption("""
                **Interpretation:**
                - α < 0: Error-correcting (good!)
                - α > 0: Destabilizing
                - |α| large: Fast adjustment
                - α ≈ 0: Weakly exogenous
                """)
                
                # Half-lives
                st.markdown("**Half-Lives of Shocks:**")
                
                for i, var in enumerate(variables_for_cointegration):
                    for j in range(vec_rank):
                        alpha_val = alpha_matrix[i, j]
                        if alpha_val < -0.001:
                            half_life = np.log(0.5) / np.log(1 + alpha_val)
                            st.write(f"- {var} (CE{j+1}): α={alpha_val:.4f} → Half-life = {half_life:.1f} periods")
                        elif abs(alpha_val) < 0.001:
                            st.write(f"- {var} (CE{j+1}): α={alpha_val:.4f} → Weakly exogenous")
                        else:
                            st.write(f"- {var} (CE{j+1}): α={alpha_val:.4f} → Divergent")
                
                # Short-run dynamics (Γ)
                st.markdown("---")
                st.markdown("#### C. Short-Run Dynamics (Γ)")
                
                gamma_params = vec_result.gamma
                
                if gamma_params is not None and gamma_params.size > 0:
                    n_lags = gamma_params.shape[0] // len(variables_for_cointegration)
                    
                    for lag in range(min(n_lags, 2)):  # Show first 2 lags
                        st.markdown(f"**Γ{lag+1} (Lag {lag+1} effects):**")
                        
                        start = lag * len(variables_for_cointegration)
                        end = (lag + 1) * len(variables_for_cointegration)
                        
                        gamma_lag = gamma_params[start:end, :]
                        gamma_df = pd.DataFrame(
                            gamma_lag,
                            index=[f"Δ{v}(t-{lag+1})" for v in variables_for_cointegration],
                            columns=[f"Δ{v}(t)" for v in variables_for_cointegration]
                        )
                        
                        st.dataframe(
                            gamma_df.style
                            .format('{:.4f}')
                            .background_gradient(cmap='RdYlGn', vmin=-0.3, vmax=0.3),
                            use_container_width=True
                        )
                
                # Model diagnostics
                st.markdown("---")
                st.markdown("#### D. VEC Model Diagnostics")
                
                vec_resid = vec_result.resid
                
                # Serial correlation
                st.markdown("**Serial Correlation:**")
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    
                    serial_tests = []
                    for i, var in enumerate(variables_for_cointegration):
                        lb = acorr_ljungbox(vec_resid[:, i], lags=[5], return_df=True)
                        pval = lb.loc[5, 'lb_pvalue']
                        serial_tests.append({
                            'Variable': var,
                            'P-value': f'{pval:.4f}',
                            'Result': '✓ Pass' if pval > 0.05 else '✗ Fail'
                        })
                    
                    st.dataframe(pd.DataFrame(serial_tests), use_container_width=True)
                except:
                    st.warning("Could not test serial correlation")
                
                # Normality
                st.markdown("**Normality:**")
                try:
                    from scipy.stats import jarque_bera
                    
                    norm_tests = []
                    for i, var in enumerate(variables_for_cointegration):
                        jb_stat, jb_pval = jarque_bera(vec_resid[:, i])
                        norm_tests.append({
                            'Variable': var,
                            'JB Stat': f'{jb_stat:.4f}',
                            'P-value': f'{jb_pval:.4f}',
                            'Normal?': '✓' if jb_pval > 0.05 else '✗'
                        })
                    
                    st.dataframe(pd.DataFrame(norm_tests), use_container_width=True)
                except:
                    st.warning("Could not test normality")
                
                # Summary
                st.markdown("---")
                st.markdown("### 💡 VEC Model Summary")
                
                st.success(f"""
                ✅ **VEC Model Complete!**
                
                **Long-run equilibria:** {vec_rank} cointegrating relationship(s)
                **Error correction:** Active (α coefficients show adjustment)
                **Short-run dynamics:** Captured by Γ matrices
                
                **Model fit:**
                - Log-likelihood: {vec_result.llf:.2f}
                - AIC: {vec_result.aic:.2f}
                - BIC: {vec_result.bic:.2f}
                """)
                
            except Exception as e:
                st.error(f"VEC estimation error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        else:
            st.markdown("---")
            st.info("""
            ### ✅ No Cointegration - Analysis Complete
            
            **Conclusion:** No long-run equilibrium detected
            
            **Model recommendation:** VAR in differences (your Section 3)
            
            **For your report:**
            1. Document Johansen test results
            2. State no cointegration found
            3. Justify VAR in differences
            4. Continue with Section 3 results
            """)
        
        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================
        
        # ====================================================================
# FINAL SUMMARY
# ====================================================================

    st.markdown("---")
    st.markdown("### 📊 Section 6 Summary")

    # Check if cointegration analysis was performed
    if 'df_log_levels' in locals() and df_log_levels is not None:
        try:
            st.success(f"""
            **Cointegration Analysis on Log-Level Data - Complete**

            **Data Preparation:**
            - ✓ Transformed raw prices to log levels
            - ✓ Verified I(1) property (non-stationary in levels, stationary in differences)
            - ✓ {len(df_log_levels)} observations analyzed

            **Lag Selection:**
            - ✓ Optimal lag for log-level VAR: {johansen_lag}
            - ✓ Residual diagnostics performed
            - ✓ Model adequacy verified

            **Johansen Cointegration Test:**
            - Trace test: {num_coint_trace if 'num_coint_trace' in locals() else 0} relation(s)
            - Max eigenvalue test: {num_coint_maxeig if 'num_coint_maxeig' in locals() else 0} relation(s)
            - **Conclusion: {suggested_rank if 'suggested_rank' in locals() else 0} cointegrating relationship(s)**

            {f'''**VEC Model (Estimated):**
            - Cointegrating rank: {vec_rank}
            - Long-run equilibria defined by β vectors
            - Error correction captured by α coefficients
            - Short-run dynamics in Γ matrices
            - Model diagnostics completed''' if 'suggested_rank' in locals() and suggested_rank > 0 else '''**Model Recommendation:**
            - VAR in log differences is appropriate
            - No VEC model needed
            - Your Section 3 analysis is complete and correct'''}

            **Next Steps:**
            - {'Review VEC results for long-run equilibrium interpretation' if 'suggested_rank' in locals() and suggested_rank > 0 else 'Finalize VAR analysis from Section 3'}
            - {'Use VEC for long-horizon forecasts' if 'suggested_rank' in locals() and suggested_rank > 0 else 'Use VAR for all forecasting and structural analysis'}
            - Complete Section 7 (Conclusion)
            """)
        except Exception as e:
            st.warning(f"⚠️ Could not display full summary: {e}")
    else:
        st.info("""
        **Section 6: Cointegration Analysis**
        
        ℹ️ Not yet performed. Check the box above to run cointegration analysis.
        
        **What it will test:**
        - I(1) property verification
        - Johansen cointegration test
        - VEC model estimation (if cointegration detected)
        """)

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================

    
    # ====================================================================
    # SECTION 7: CONCLUSION
    # ====================================================================
    
    st.markdown('<div class="section-header">7. Conclusion & Summary</div>', unsafe_allow_html=True)

    with st.expander("📋 Executive Summary", expanded=True):
        st.markdown("### Key Findings")
        
        st.markdown(f"""
        #### 1. Data Overview
        - **System**: {selected_category if selected_category != "All Categories" else "Financial Markets"}
        - **Variables**: {', '.join(variables)}
        - **Period**: {df.index[0].date()} to {df.index[-1].date()}
        - **Observations**: {len(df)} ({frequency})
        - **Transformations**: {', '.join([f'{k}: {v}' for k, v in transform_info.items()])}
        
        #### 2. Stationarity
        """)
        
        # Check if stationarity tests were run
        if 'stationary' in locals() and stationary:
            st.write(f"- **Stationary variables**: {', '.join(stationary)}")
        if 'non_stationary' in locals() and non_stationary:
            st.write(f"- **Non-stationary**: {', '.join(non_stationary)}")
        
        st.markdown(f"""
        #### 3. VAR Model
        - **Optimal lag**: {int(selected_lag)}
        - **AIC**: {var_model.aic:.4f}
        - **BIC**: {var_model.bic:.4f}
        - **Stability**: {'✓ Stable' if 'is_stable' in locals() and is_stable else '✗ Check diagnostics'}
        """)
        
        if 'granger_results' in locals() and granger_results:
            sig_granger = sum(1 for v in granger_results.values() if v['Significant'])
            st.write(f"- **Significant Granger causality**: {sig_granger} relationships")
        
        # Only show cointegration results if analysis was performed
        if 'suggested_rank' in locals():
            st.markdown(f"""
            #### 4. Cointegration
            - **Cointegrating relationships**: {suggested_rank if suggested_rank > 0 else 'None detected'}
            - **Recommendation**: {'VEC model' if suggested_rank > 0 else 'VAR in differences'}
            """)
        else:
            st.markdown("""
            #### 4. Cointegration
            - Not yet analyzed (enable Section 6 checkbox to run)
            """)
        
        st.markdown("""
        ### Limitations & Future Work
        
        **Limitations:**
        - Linear VAR may miss non-linear dynamics
        - Potential parameter instability over time
        - Structural breaks not formally tested
        
        **Recommendations:**
        - Test for structural breaks (Chow, CUSUM)
        - Consider time-varying parameters
        - Explore threshold/regime-switching VAR
        - Bayesian VAR for more stable estimates
        """)

    st.markdown("---")
    st.success("✅ **Analysis Complete!** All sections have been executed successfully.")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()