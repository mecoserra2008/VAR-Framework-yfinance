# ============================================================================
# statistical_tests.py - Statistical Tests Module
# ============================================================================
"""
This module handles:
- Augmented Dickey-Fuller (ADF) tests
- KPSS stationarity tests
- Seasonality detection
- Granger causality matrix tests
"""

import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose

# ============================================================================
# STANDARD STATIONARITY TESTS
# ============================================================================

def perform_adf_test(series, name, regression='ct'):
    """Augmented Dickey-Fuller test with proper specification"""
    try:
        result = adfuller(series.dropna(), maxlag=None, regression=regression, autolag='AIC')
        return {
            'Variable': name,
            'Regression': regression,
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Observations': result[3],
            'Critical Value (1%)': result[4]['1%'],
            'Critical Value (5%)': result[4]['5%'],
            'Critical Value (10%)': result[4]['10%'],
            'Stationary (5%)': result[1] < 0.05
        }
    except Exception as e:
        return {
            'Variable': name,
            'Regression': regression,
            'ADF Statistic': np.nan,
            'p-value': np.nan,
            'Lags Used': np.nan,
            'Observations': len(series),
            'Critical Value (1%)': np.nan,
            'Critical Value (5%)': np.nan,
            'Critical Value (10%)': np.nan,
            'Stationary (5%)': False
        }

def perform_kpss_test(series, name, regression='ct'):
    """KPSS test as complement to ADF"""
    try:
        result = kpss(series.dropna(), regression=regression, nlags='auto')
        return {
            'Variable': name,
            'KPSS Statistic': result[0],
            'p-value': result[1],
            'Critical Value (10%)': result[3]['10%'],
            'Critical Value (5%)': result[3]['5%'],
            'Critical Value (1%)': result[3]['1%'],
            'Stationary (5%)': result[1] > 0.05  # KPSS null is stationarity
        }
    except Exception as e:
        return {
            'Variable': name,
            'KPSS Statistic': np.nan,
            'p-value': np.nan,
            'Critical Value (10%)': np.nan,
            'Critical Value (5%)': np.nan,
            'Critical Value (1%)': np.nan,
            'Stationary (5%)': False
        }

# ============================================================================
# SEASONALITY DETECTION
# ============================================================================

def detect_seasonality(series, freq=12):
    """Detect seasonality in time series"""
    try:
        if len(series) < 2 * freq:
            return False, "Insufficient data for seasonality test"
        
        decomposition = seasonal_decompose(series.dropna(), model='additive', period=freq, extrapolate_trend='freq')
        seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.resid + decomposition.seasonal)
        
        has_seasonality = seasonal_strength > 0.3
        return has_seasonality, f"Seasonal strength: {seasonal_strength:.3f}"
    except:
        return False, "Could not detect seasonality"

# ============================================================================
# GRANGER CAUSALITY MATRIX
# ============================================================================

def test_granger_causality_matrix(data, variables, max_lag=4):
    """Test Granger causality for all pairs"""
    results = {}
    for var1 in variables:
        for var2 in variables:
            if var1 != var2:
                try:
                    test = grangercausalitytests(data[[var2, var1]], maxlag=max_lag, verbose=False)
                    p_values = [test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    min_p_value = min(p_values)
                    best_lag = p_values.index(min_p_value) + 1
                    results[f"{var1} → {var2}"] = {
                        'p-value': min_p_value,
                        'best_lag': best_lag,
                        'Significant': min_p_value < 0.05
                    }
                except:
                    results[f"{var1} → {var2}"] = {
                        'p-value': np.nan,
                        'best_lag': np.nan,
                        'Significant': False
                    }
    return results
