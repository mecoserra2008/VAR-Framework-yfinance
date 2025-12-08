# VAR & Cointegration Analysis Framework

**ISEG - Instituto Superior de Economia e Gestão**
**Macroeconometrics II - Team Work**

## Overview

This is a Streamlit app created by Américo Serra and [Gulnora Nizomova](https://github.com/GulnoraN) to fulfill the needs of the modelling structure proposed by Gabriel Zsurkis, professor of Macroeconometrics II in Msc in Applied Econometrics and Forecasting at ISEG. Fetches data from Yahoo Finance and performs econometric testing, model estimation, and forecasting.

## Requirements

- Python 3.11
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0,<2.3.0` - Data manipulation
- `numpy>=1.24.0,<2.0.0` - Numerical computations
- `yfinance>=0.2.28` - Yahoo Finance data fetching
- `plotly>=5.17.0` - Interactive visualizations
- `scipy>=1.11.0` - Statistical computations
- `statsmodels>=0.14.0` - Econometric models
- `scikit-learn>=1.3.0` - Machine learning utilities
- `pyyaml>=6.0` - YAML configuration parsing
- `matplotlib>=3.7.0` - Plotting backend for pandas styling

## Configuration

### Asset Configuration

Edit `assets.yaml` to define financial instruments for analysis. Structure:

```yaml
asset_categories:
  category_name:
    name: "Display Name"
    symbols:
      - symbol: "TICKER"
        name: "Asset Name"
        category: "Category"
        description: "Description"
```

**Note:** The application includes predefined US Treasury yields (`^IRX`, `^FVX`, `^TNX`, `^TYX`) in addition to assets defined in `assets.yaml`.

### Path Configuration

Update line 29 in `data_loader.py` if `assets.yaml` is not in the default location:

```python
yaml_path = Path(r"path/to/your/assets.yaml")
```

## Running the Application

```bash
streamlit run app.py
```

The application opens in your default browser (typically `http://localhost:8501`).

## Workflow

### 1. Data Selection

**Sidebar Configuration:**

- **Variables:** Select 3-4 financial instruments from dropdown
- **Date Range:** Set start and end dates for historical data
- **Frequency:** Choose daily, weekly, or monthly observations
- **Transformation:** Select data transformation method:
  - Level (no transformation)
  - Log transformation
  - First difference
  - Log difference (returns)

**Validation:**
- Minimum 20 observations required
- Data automatically cleaned for missing values
- Failed symbol fetches are reported separately

### 2. Stationarity Testing

**Augmented Dickey-Fuller (ADF) Test:**
- H₀: Series has unit root (non-stationary)
- Reject H₀ if p-value < 0.05
- Regression specification: constant + trend (`ct`)

**KPSS Test:**
- H₀: Series is stationary
- Reject H₀ if p-value < 0.05
- Complementary to ADF test

**Transformation Recommendations:**
- Automatically suggests transformations based on test results
- Log-differencing recommended for positive non-stationary series
- First differencing recommended for series with negative values

### 3. Lag Order Selection

**Information Criteria:**
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- HQIC (Hannan-Quinn Information Criterion)
- FPE (Final Prediction Error)

**Interpretation:**
- Lower values indicate better fit
- BIC tends to select more parsimonious models
- AIC may select higher lags

**Recommendation:** Select lag order where multiple criteria agree, balancing model fit and parsimony.

### 4. VAR Model Estimation

**Model Specification:**
- VAR(p) model with selected lag order
- Includes constant term
- Maximum likelihood estimation

**Output:**
- Coefficient estimates for each equation
- Standard errors and t-statistics
- P-values for significance testing
- R-squared and adjusted R-squared per equation

### 5. Model Diagnostics

#### Stability Analysis

**Eigenvalue Table:**
- Lists eigenvalues, roots, and inverse roots
- Model stable if all eigenvalue moduli < 1
- Unstable models produce unreliable forecasts

**Interpretation:**
- ✓ YES: Stable eigenvalue (modulus < 1)
- ✗ NO: Unstable eigenvalue (modulus ≥ 1)

#### Residual Diagnostics

**Serial Correlation (Portmanteau Test):**
- H₀: No serial correlation in residuals
- Cannot reject H₀ if p-value > 0.05
- Tests multiple lags simultaneously

**Normality (Jarque-Bera Test):**
- H₀: Residuals normally distributed
- Cannot reject H₀ if p-value > 0.05
- Tests each equation separately

**Heteroskedasticity (White Test):**
- H₀: Homoskedastic residuals
- Cannot reject H₀ if p-value > 0.05
- Important for inference validity

### 6. Granger Causality Analysis

**Standard Granger Causality:**
- H₀: Variable X does NOT Granger-cause variable Y
- Tests whether past values of X predict Y
- Reject H₀ if p-value < 0.05

**Instantaneous Causality:**
- Tests contemporaneous correlation in residuals
- H₀: No instantaneous causality between variables
- Based on residual covariance matrix

**Interpretation:**
- Significant result: Evidence of predictive relationship
- Not causal in strict sense - shows temporal precedence
- Direction matters: X → Y differs from Y → X

### 7. Impulse Response Functions (IRF)

**Definition:** Response of variable Y to one-unit shock in variable X over time.

**Types:**
- **Standard IRF:** Non-orthogonalized shocks
- **Orthogonalized IRF:** Uses Cholesky decomposition (ordering-dependent)

**Confidence Intervals:**
- Default: 95% confidence bands
- Asymptotic standard errors
- Shaded region shows uncertainty

**Interpretation:**
- Positive response: Y increases following shock to X
- Negative response: Y decreases following shock to X
- Persistence: How many periods effect lasts
- Zero line crossing: Effect becomes insignificant

### 8. Forecasting

**Forecast Output:**
- Point forecasts for each variable
- Confidence intervals (default 95%)
- MSE (Mean Squared Error) matrices per horizon
- Forecast horizon user-specified (typically 10 periods)

**Interpretation:**
- Point forecast: Expected value at each horizon
- Confidence bands: Wider bands = greater uncertainty
- MSE increases with forecast horizon

### 9. Cointegration Analysis

**Purpose:** Test for long-run equilibrium relationships among non-stationary variables.

**Johansen Cointegration Test:**

**Deterministic Trend Specifications:**
- Model 0: No deterministic trend
- Model 1: Restricted constant
- Model 2: Unrestricted constant
- Model 3: Restricted trend
- Model 4: Unrestricted trend

**Test Statistics:**
- **Trace Statistic:** Tests H₀: at most r cointegrating relationships
- **Max Eigenvalue Statistic:** Tests H₀: exactly r cointegrating relationships

**Interpretation:**
- Reject H₀ if test statistic > critical value
- Number of cointegrating vectors = r
- r = 0: No cointegration
- r ≥ 1: Long-run equilibrium exists

**Eigenvalues and Eigenvectors:**
- Eigenvalues: Strength of cointegrating relationships
- Eigenvectors: Weights in cointegrating combinations
- First eigenvector = strongest relationship

### 10. Vector Error Correction Model (VECM)

**Specification:** Applied when cointegration detected (r > 0).

**Model Components:**
- Error correction term (α × β')
- Short-run dynamics (lagged differences)
- α: Speed of adjustment to equilibrium
- β: Long-run cointegrating vector

**Output:**
- α coefficients: Adjustment speeds
- β coefficients: Long-run relationships
- Short-run coefficient estimates
- Standard errors and significance tests

**Interpretation:**
- Negative α: System corrects toward equilibrium
- Larger |α|: Faster adjustment
- β coefficients: Long-run elasticities

## Results Interpretation Summary

### Stationarity
- **Stationary:** Safe to estimate VAR in levels
- **Non-stationary:** Apply transformation or check cointegration

### Model Stability
- **Stable:** Proceed with inference and forecasting
- **Unstable:** Re-specify model (different lag, transformation, or variables)

### Granger Causality
- **Significant:** Past values of X help predict Y
- **Not significant:** No predictive relationship

### Cointegration
- **r = 0:** No long-run relationship, use VAR in differences
- **r ≥ 1:** Long-run equilibrium exists, use VECM

### IRF
- **Magnitude:** Size of response
- **Sign:** Direction of response (positive/negative)
- **Persistence:** Duration of effect
- **Confidence bands:** Statistical significance

## Downloading Results

### Method 1: Screenshot
- Use browser screenshot tools for charts
- Right-click plots → "Save image as..."

### Method 2: Data Export
- Dataframes displayed in tables can be copied
- Select table → Copy → Paste to Excel/CSV

### Method 3: Programmatic Export

Add to `app.py` after relevant analysis sections:

```python
# Export forecast results
forecast_df.to_csv("forecast_results.csv")

# Export test results
results_df.to_csv("test_results.csv")

# Export IRF data
irf_df = pd.DataFrame(irf_results['irf'])
irf_df.to_csv("irf_results.csv")
```

### Method 4: Plotly Export
- Hover over plot → Camera icon → Download as PNG
- Click plot → Hamburger menu → Export to static image

## File Structure

```
.
├── app.py                  # Main Streamlit application
├── assets.yaml             # Financial instrument configuration
├── causality_irf.py        # Granger causality and IRF functions
├── cointegration.py        # Johansen test and VECM functions
├── data_loader.py          # Yahoo Finance data fetching and transformation
├── statistical_tests.py    # ADF, KPSS, seasonality tests
├── var_model.py            # VAR stability, diagnostics, forecasting
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Module Descriptions

### app.py
Main application orchestrating workflow: data selection, stationarity testing, VAR/VECM estimation, diagnostics, causality analysis, IRF, and forecasting.

### data_loader.py
- Loads asset configurations from YAML
- Fetches historical data from Yahoo Finance
- Handles batch retrieval for long time periods
- Applies data transformations (log, difference, returns)
- Validates data quality and completeness

### statistical_tests.py
- Augmented Dickey-Fuller unit root test
- KPSS stationarity test
- Seasonality detection
- Granger causality matrix computation

### var_model.py
- VAR model stability analysis (eigenvalues, roots)
- Coefficient significance testing
- Multi-step forecasting with confidence intervals
- MSE matrix computation

### causality_irf.py
- Instantaneous causality tests
- Comprehensive Granger causality analysis
- Impulse response function computation
- IRF confidence intervals and plotting

### cointegration.py
- Simplified ADF/KPSS tests (matching R defaults)
- Johansen cointegration test
- Multiple deterministic trend specifications
- Eigenvector and eigenvalue extraction

## Common Issues

### Data Loading Fails
- Verify internet connection
- Check ticker symbols in `assets.yaml` are valid
- Yahoo Finance may have API rate limits - reduce number of symbols or date range
- Update `assets.yaml` path in `data_loader.py:29`

### Insufficient Observations
- Extend date range
- Use daily frequency instead of weekly/monthly
- Select assets with longer trading history

### Model Instability
- Reduce lag order
- Apply different transformation
- Remove problematic variables
- Check for outliers or structural breaks

### No Cointegration Detected
- Variables may not have long-run relationship
- Try different deterministic trend specification
- Verify all variables are I(1) before testing

## Technical Notes

### Data Frequency
- **Daily:** Most granular, largest sample size
- **Weekly:** Friday closing prices (resample rule: `W-FRI`)
- **Monthly:** Last trading day of month (resample rule: `M`)

### Transformation Guidelines
- **Prices:** Typically non-stationary, use log differences (returns)
- **Interest rates:** May be stationary in levels or first difference
- **Exchange rates:** Usually non-stationary, use log differences
- **Volatility indices:** Check stationarity before transforming

### Model Selection
- **Stationary variables:** VAR in levels
- **Non-stationary, no cointegration:** VAR in differences
- **Non-stationary with cointegration:** VECM

### Sample Size Requirements
- Minimum: 20 observations (enforced)
- Recommended: 50+ observations for reliable inference
- Rule of thumb: 10-20 observations per parameter estimated

## References

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.

## License

See LICENSE file for details.

## Support

For issues related to:
- **Application bugs:** Check data inputs and configuration
- **Statistical methodology:** Consult econometrics references above
- **Yahoo Finance data:** Verify ticker symbols on finance.yahoo.com
