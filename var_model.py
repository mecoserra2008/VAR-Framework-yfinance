# ============================================================================
# var_model.py - VAR Model Estimation and Diagnostics Module
# ============================================================================
"""
This module handles:
- VAR model stability analysis
- Coefficient significance testing
- Forecasting with confidence intervals
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats

# ============================================================================
# STABILITY ANALYSIS
# ============================================================================

def display_combined_stability_table(var_model, variables):
    """Display a single combined table with eigenvalues, roots, inverse roots and stability status"""
    try:
        # Convert params to numpy array
        params_array = var_model.params.values
        
        k = len(variables)  # number of variables
        p = var_model.k_ar  # lag order
        
        # Build proper companion matrix
        # Get coefficient matrices for each lag
        coef_matrices = []
        for i in range(p):
            # Extract coefficients for lag i (skip constant if present)
            start_idx = i * k
            end_idx = (i + 1) * k
            coef_matrices.append(params_array[start_idx:end_idx, :].T)
        
        # Build companion matrix (k*p x k*p)
        companion = np.zeros((k*p, k*p))
        companion[:k, :] = np.hstack(coef_matrices)
        if p > 1:
            companion[k:, :-k] = np.eye(k*(p-1))
        
        # Calculate eigenvalues from proper companion matrix
        eigenvalues = np.linalg.eigvals(companion)
        moduli = np.abs(eigenvalues)
        
        # Get roots from the model (these are already calculated correctly)
        roots = var_model.roots
        roots_moduli = np.abs(roots)
        
        # Calculate inverse roots
        inverse_roots = 1 / roots
        
        # Create combined dataframe
        stability_data = []
        for i in range(len(eigenvalues)):
            stability_data.append({
                'Index': i + 1,
                'Eigenvalue (Real)': np.real(eigenvalues[i]),
                'Eigenvalue (Imag)': np.imag(eigenvalues[i]),
                'Eigenvalue |λ|': moduli[i],
                'Root (Real)': np.real(roots[i]),
                'Root (Imag)': np.imag(roots[i]),
                'Root |1/λ|': roots_moduli[i],
                'Inverse Root (Real)': np.real(inverse_roots[i]),
                'Inverse Root (Imag)': np.imag(inverse_roots[i]),
                'Stable?': '✓ YES' if moduli[i] < 1 else '✗ NO'
            })
        
        stability_df = pd.DataFrame(stability_data)
        is_stable = all(moduli < 1)
        
        return is_stable, stability_df
        
    except Exception as e:
        st.error(f"Error in stability analysis: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# ============================================================================
# COEFFICIENT SIGNIFICANCE
# ============================================================================

def analyze_coefficient_significance(var_result, significance_level=0.05):
    """Analyze which coefficients are insignificant"""
    params = var_result.params
    pvalues = var_result.pvalues
    
    insignificant_list = []
    n_insignificant = 0
    
    for eq in pvalues.columns:
        for param in pvalues.index:
            if 'const' not in param.lower():  # Don't consider constant term
                if pvalues.loc[param, eq] > significance_level:
                    n_insignificant += 1
                    insignificant_list.append({
                        'Equation': eq,
                        'Parameter': param,
                        'Coefficient': params.loc[param, eq],
                        'p-value': pvalues.loc[param, eq],
                        'Significant': 'No'
                    })
    
    return n_insignificant, insignificant_list

# ============================================================================
# FORECASTING WITH CONFIDENCE INTERVALS
# ============================================================================

def forecast_with_mse_and_ci(var_model, df, steps=10, alpha=0.05):
    """Generate forecasts with MSE matrices and confidence intervals"""
    try:
        # Generate forecast
        forecast_result = var_model.forecast(df.values[-var_model.k_ar:], steps=steps)
        
        # Get MSE matrices
        mse_matrices = var_model.mse(steps=steps)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(
            forecast_result,
            columns=df.columns,
            index=pd.date_range(start=df.index[-1], periods=steps+1, freq=df.index.freq)[1:]
        )
        
        # Calculate confidence intervals
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bounds = {}
        upper_bounds = {}
        
        for i, var in enumerate(df.columns):
            lower = []
            upper = []
            for h in range(steps):
                std_error = np.sqrt(mse_matrices[h][i, i])
                lower.append(forecast_result[h, i] - z_score * std_error)
                upper.append(forecast_result[h, i] + z_score * std_error)
            
            lower_bounds[var] = lower
            upper_bounds[var] = upper
        
        return forecast_df, lower_bounds, upper_bounds, mse_matrices
        
    except Exception as e:
        st.error(f"Error in forecasting: {e}")
        return None, None, None, None
