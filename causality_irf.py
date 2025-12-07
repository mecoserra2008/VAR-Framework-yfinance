# ============================================================================
# causality_irf.py - Granger Causality and IRF Module
# ============================================================================
"""
This module handles:
- Instantaneous causality tests
- Comprehensive Granger causality tests
- Impulse Response Functions (IRF) with confidence intervals
- Plotting IRF with confidence bands
"""

import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import chi2
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# INSTANTANEOUS CAUSALITY
# ============================================================================

def test_instantaneous_causality(var_result, variables, significance_level=0.05):
    """
    Test for instantaneous causality using Wald test on residual covariance matrix.
    Tests H0: σ_ij = 0 for all i ≠ j
    """
    results = {}
    sigma_u = var_result.sigma_u
    
    # CRITICAL FIX: Convert to numpy array if it's a DataFrame
    if isinstance(sigma_u, pd.DataFrame):
        sigma_u = sigma_u.values
    
    n_obs = var_result.nobs
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                sigma_ij = sigma_u[i, j]
                sigma_ii = sigma_u[i, i]
                sigma_jj = sigma_u[j, j]
                
                test_stat = n_obs * (sigma_ij ** 2) / (sigma_ii * sigma_jj)
                p_value = 1 - chi2.cdf(test_stat, df=1)
                critical_value = chi2.ppf(1 - significance_level, df=1)
                significant = p_value < significance_level
                
                results[f"{var1} <-> {var2}"] = {
                    'Covariance': sigma_ij,
                    'Test Statistic': test_stat,
                    'p-value': p_value,
                    'Critical Value (95%)': critical_value,
                    'Significant': significant,
                    'Interpretation': f"{'Reject H0' if significant else 'Cannot reject H0'}: {'Evidence of' if significant else 'No'} instantaneous causality"
                }
    
    return results

# ============================================================================
# COMPREHENSIVE GRANGER CAUSALITY
# ============================================================================

def comprehensive_granger_causality(var_result, variables, maxlag=None, significance_level=0.05):
    """
    Perform comprehensive Granger causality tests using Wald statistics.
    """
    if maxlag is None:
        maxlag = var_result.k_ar
    
    st.markdown("#### 🔍 Standard Granger Causality Tests")
    st.markdown("**H₀:** Variable X does NOT Granger-cause variable Y")
    
    granger_results = {}
    granger_data = []
    
    data_for_test = var_result.endog
    df_test = pd.DataFrame(data_for_test, columns=variables)
    
    for cause_var in variables:
        for effect_var in variables:
            if cause_var != effect_var:
                try:
                    test_result = grangercausalitytests(
                        df_test[[effect_var, cause_var]], 
                        maxlag=maxlag, 
                        verbose=False
                    )
                    
                    wald_stat = test_result[maxlag][0]['ssr_ftest'][0]
                    p_value = test_result[maxlag][0]['ssr_ftest'][1]
                    significant = p_value < significance_level
                    
                    granger_results[f"{cause_var} → {effect_var}"] = {
                        'Wald Statistic': wald_stat,
                        'p-value': p_value,
                        'Significant': significant
                    }
                    
                    granger_data.append({
                        'Cause (X)': cause_var,
                        'Effect (Y)': effect_var,
                        'Wald Stat': wald_stat,
                        'p-value': p_value,
                        'Significant (5%)': '✓ YES' if significant else '✗ NO',
                        'Interpretation': f"X {'does' if significant else 'does NOT'} Granger-cause Y"
                    })
                    
                except Exception as e:
                    st.warning(f"Could not test {cause_var} → {effect_var}: {e}")
    
    if granger_data:
        granger_df = pd.DataFrame(granger_data)
        st.dataframe(granger_df.style.format({
            'Wald Stat': '{:.4f}',
            'p-value': '{:.4f}'
        }).apply(lambda x: ['background-color: #d4edda' if v == '✓ YES' 
                            else 'background-color: #f8d7da' if v == '✗ NO' 
                            else '' for v in x], subset=['Significant (5%)']),
                    use_container_width=True)
        
        n_significant = sum(1 for r in granger_results.values() if r['Significant'])
        st.info(f"**Summary:** {n_significant} out of {len(granger_results)} relationships show significant Granger causality at 5% level")
    
    return granger_results

# ============================================================================
# IRF WITH CONFIDENCE INTERVALS
# ============================================================================

def compute_irf_with_ci(var_result, steps=10, alpha=0.05, method='asymptotic'):
    """
    Compute Impulse Response Functions with confidence intervals.
    """
    irf = var_result.irf(steps)
    irf_se = irf.stderr()
    
    z_critical = stats.norm.ppf(1 - alpha/2)
    lower_bound = irf.irfs - z_critical * irf_se
    upper_bound = irf.irfs + z_critical * irf_se
    
    return {
        'irf': irf.irfs,
        'lower': lower_bound,
        'upper': upper_bound,
        'stderr': irf_se,
        'orth_irf': irf.orth_irfs if hasattr(irf, 'orth_irfs') else None
    }

def plot_irf_with_ci(var_result, variables, steps=10, impulse_var=None, response_var=None, 
                     orthogonalized=False, alpha=0.05):
    """
    Plot Impulse Response Functions with confidence intervals.
    """
    irf_ci = compute_irf_with_ci(var_result, steps=steps, alpha=alpha)
    
    if orthogonalized and irf_ci['orth_irf'] is not None:
        irf_data = irf_ci['orth_irf']
        title_suffix = " (Orthogonalized)"
    else:
        irf_data = irf_ci['irf']
        title_suffix = ""
    
    n_vars = len(variables)
    
    if impulse_var and response_var:
        impulse_idx = variables.index(impulse_var)
        response_idx = variables.index(response_var)
        
        fig = go.Figure()
        periods = list(range(steps))
        
        fig.add_trace(go.Scatter(
            x=periods + periods[::-1],
            y=np.concatenate([irf_ci['upper'][:, response_idx, impulse_idx],
                            irf_ci['lower'][:, response_idx, impulse_idx][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{int((1-alpha)*100)}% CI',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=periods,
            y=irf_data[:, response_idx, impulse_idx],
            mode='lines',
            name='IRF',
            line=dict(color='darkblue', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title=f"Impulse Response: {impulse_var} → {response_var}{title_suffix}",
            xaxis_title="Periods",
            yaxis_title="Response",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    else:
        fig = make_subplots(
            rows=n_vars, cols=n_vars,
            subplot_titles=[f"{variables[j]} → {variables[i]}" 
                          for i in range(n_vars) for j in range(n_vars)],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        periods = list(range(steps))
        
        for i in range(n_vars):
            for j in range(n_vars):
                row = i + 1
                col = j + 1
                
                fig.add_trace(go.Scatter(
                    x=periods + periods[::-1],
                    y=np.concatenate([irf_ci['upper'][:, i, j],
                                    irf_ci['lower'][:, i, j][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row, col=col)
                
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=irf_data[:, i, j],
                    mode='lines',
                    line=dict(color='darkblue', width=1.5),
                    showlegend=False,
                    name=f"{variables[j]} → {variables[i]}"
                ), row=row, col=col)
                
                fig.add_hline(y=0, line_dash="dash", line_color="red", 
                            opacity=0.3, row=row, col=col)
        
        fig.update_layout(
            title_text=f"Impulse Response Functions{title_suffix} with {int((1-alpha)*100)}% Confidence Intervals",
            height=300 * n_vars,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Periods")
        fig.update_yaxes(title_text="Response")
        
        return fig
