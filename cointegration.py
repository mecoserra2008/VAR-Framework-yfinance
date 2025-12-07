# ============================================================================
# cointegration.py - Johansen Test and VEC Model Module
# ============================================================================
"""
This module handles:
- Specialized ADF/KPSS tests for cointegration analysis (matching R defaults)
- Johansen cointegration test
- VEC model estimation and diagnostics
"""

import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, kpss

# ============================================================================
# SPECIALIZED STATIONARITY TESTS FOR COINTEGRATION
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
# JOHANSEN COINTEGRATION TEST
# ============================================================================

def perform_johansen_test(data, det_order=0, k_ar_diff=1):
    """Perform Johansen cointegration test"""
    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        
        return {
            'trace_stat': result.lr1,
            'trace_crit_90': result.cvt[:, 0],
            'trace_crit_95': result.cvt[:, 1],
            'trace_crit_99': result.cvt[:, 2],
            'max_eig_stat': result.lr2,
            'max_eig_crit_90': result.cvm[:, 0],
            'max_eig_crit_95': result.cvm[:, 1],
            'max_eig_crit_99': result.cvm[:, 2],
            'eigenvalues': result.eig,
            'eigenvectors': result.evec
        }
    except Exception as e:
        st.error(f"Johansen test error: {e}")
        return None
