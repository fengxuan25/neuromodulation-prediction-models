# ACh Ridge Regression Analysis - Python Translation

This repository contains a Python translation of MATLAB code for analyzing acetylcholine (ACh) signals using ridge regression. The analysis predicts ACh signals from behavioral variables including running speed, position, licking events, and reward delivery.

## Features

- **Ridge regression modeling** with cross-validation for optimal regularization
- **Multiple predictor types**: speed (linear & quadratic), position binning, temporal event kernels
- **Signal deconvolution** using bi-exponential kernels
- **Comprehensive visualization** including scatter plots, time series, variance decomposition, and temporal kernels
- **Multi-session analysis** to compare results across recording sessions
- **Jupyter notebook friendly** with example workflows

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy scipy scikit-learn matplotlib jupyter
```

2. Start Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `ACh_Analysis_Tutorial.ipynb` to get started!

## Quick Start

### Single Session Analysis

```python
from predict_ach_glm import analyze_ach_single_session

# Run analysis on a single .mat file
results = analyze_ach_single_session(
    mat_file_path='path/to/your/data.mat',
    idx_start=0,           # Optional: starting index
    idx_end=10000,         # Optional: ending index
    vel_threshold=0.5,     # cm/s threshold
    include_position=True, # Include position predictors
    make_plots=True        # Generate plots
)

# Access results
print(f"R² = {results['var_explained']:.2%}")
print("Variance contributions:", results['var_contributions'])
```

### Multiple Session Analysis

```python
from predict_ach_glm import analyze_multiple_sessions

session_files = [
    'session1.mat',
    'session2.mat',
    'session3.mat'
]

all_results = analyze_multiple_sessions(
    session_files=session_files,
    vel_threshold=0.5,
    include_position=False
)
```

## Data Format

Your `.mat` file should contain the following variables:

| Variable | Description | Shape |
|----------|-------------|-------|
| `v2` | Running speed (cm/s) | (n_samples,) |
| `licks2` | Binary licking events (0 or 1) | (n_samples,) |
| `reward2` | Binary reward delivery (0 or 1) | (n_samples,) |
| `y2_norm` | Normalized position on track (0-1) | (n_samples,) |
| `Fc_sp` | Fluorescence signal (ACh sensor) | (n_samples,) |

## Key Parameters

### Analysis Parameters

- **`vel_threshold`** (default: 0.5): Minimum speed (cm/s) for including data points in analysis
- **`dt`** (default: 0.1): Sampling interval in seconds
- **`maxlag`** (default: 20 samples = 2s): Maximum temporal lag for event predictors
- **`nbins`** (default: 20): Number of bins for position encoding
- **`include_position`** (default: True): Whether to include position as a predictor

### Cross-Validation Parameters

- **`lambda_vals`** (default: `np.logspace(-2, 2, 10)`): Range of regularization parameters
- **`k`** (default: 10): Number of folds for cross-validation

### Deconvolution Parameters

- **`tau_on`** (default: 0.06): Rise time constant (seconds)
- **`tau_off`** (default: 1.0): Decay time constant (seconds)
- **`sampling_rate`** (default: 10 Hz): Data sampling rate

## What the Analysis Does

### 1. Data Preprocessing
- Loads .mat file with behavioral and fluorescence data
- Deconvolves fluorescence signal using bi-exponential kernel
- Applies velocity threshold to filter out low-speed periods

### 2. Design Matrix Construction
The model uses these predictors:
- **Speed**: Linear and quadratic terms
- **Position**: One-hot encoded bins along the track
- **Events**: Reward and licking with temporal lags (0-2 seconds)

### 3. Ridge Regression
- Performs k-fold cross-validation to find optimal regularization (lambda)
- Fits model on filtered data (above velocity threshold)
- Evaluates overall model performance (R²)

### 4. Variance Decomposition
- Calculates how much variance each predictor group explains
- Provides partial R² for each group (drop-one analysis available)

### 5. Visualization
Generates comprehensive plots including:
- Actual vs predicted scatter plot
- Time series comparison with shaded low-speed regions
- Variance contributions by predictor group
- Speed tuning curve
- Position coefficients
- Temporal kernels for events

## Example Outputs

### Variance Explained
```
Variance explained by ridge model = 45.23%
  Speed:    12.34%
  Position: 18.76%
  Reward:   8.91%
  Licking:  5.22%
```

### Model Performance
- **R² values** typically range from 0.3 to 0.7 depending on data quality
- **Temporal kernels** show how ACh responds to events over time
- **Speed tuning** reveals relationship between movement and ACh

## Differences from MATLAB Version

### Improvements
- Uses scikit-learn's Ridge regression (more stable than manual implementation)
- Better handling of cross-validation with `KFold`
- More Pythonic code structure with reusable functions
- Enhanced error handling and data validation

### Implementation Notes
- The deconvolution function uses a simplified Wiener filtering approach
- Some plotting styles may differ slightly from MATLAB
- Random state set for reproducibility (can be adjusted)

## Troubleshooting

### Common Issues

**"KeyError: 'v2'" or similar**
- Check that your .mat file variable names match expected names
- Use `scipy.io.loadmat('file.mat').keys()` to see available variables

**"Memory Error"**
- Reduce the data size by specifying `idx_start` and `idx_end`
- Consider downsampling your data before analysis

**Poor model performance (low R²)**
- Try adjusting `vel_threshold` (higher may improve R²)
- Check for NaN or infinite values in your data
- Ensure your data has sufficient variability

**Slow cross-validation**
- Reduce number of lambda values: `lambda_vals = np.logspace(-2, 2, 5)`
- Reduce number of folds: `k=5` instead of `k=10`
- Use a smaller subset of data for initial testing

## Functions Reference

### Main Analysis Functions

#### `analyze_ach_single_session()`
Complete analysis pipeline for a single recording session.

**Returns**: Dictionary with model, predictions, variance explained, and plotting data

#### `analyze_multiple_sessions()`
Batch analysis across multiple sessions with summary visualizations.

**Returns**: List of results dictionaries for each session

### Utility Functions

#### `build_design_matrix()`
Constructs the predictor matrix from behavioral variables.

#### `ridge_cross_validation()`
Finds optimal lambda through k-fold cross-validation.

#### `deconvolve_grabACh_biExp()`
Deconvolves fluorescence signal using bi-exponential kernel.

#### `compute_partial_variance()`
Calculates variance explained by each predictor group.

#### `plot_results()`
Generates comprehensive visualization of analysis results.

## Citation

If you use this code in your research, please cite the original methods and your work appropriately.

## License

This is a translation of research code. Please respect any original licensing terms.

## Support

For questions or issues:
1. Check the tutorial notebook for examples
2. Review the function docstrings for parameter details
3. Ensure your data format matches the expected structure

## Contributing

Contributions are welcome! Areas for improvement:
- Additional deconvolution methods
- More sophisticated regularization (elastic net, etc.)
- GPU acceleration for large datasets
- Additional visualization options
