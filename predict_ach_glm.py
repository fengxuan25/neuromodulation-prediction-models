"""
Ridge Regression Analysis for Acetylcholine (ACh) Prediction
Translated from MATLAB to Python for Jupyter Notebook

This script performs ridge regression to predict ACh signals from behavioral variables
including running speed, position, licking, and reward events.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')


def deconvolve_grabACh_biExp(Fc, sampling_rate, tau_on, tau_off):
    """
    Deconvolve GrabACh fluorescence signal using bi-exponential kernel
    
    Parameters:
    -----------
    Fc : array-like
        Fluorescence signal
    sampling_rate : float
        Sampling rate in Hz
    tau_on : float
        Rise time constant in seconds
    tau_off : float
        Decay time constant in seconds
    
    Returns:
    --------
    Fc_deconv : numpy array
        Deconvolved signal
    """
    dt = 1.0 / sampling_rate
    
    # Create bi-exponential kernel
    kernel_duration = 5 * tau_off  # 5 time constants
    t_kernel = np.arange(0, kernel_duration, dt)
    
    # Bi-exponential: (1 - exp(-t/tau_on)) * exp(-t/tau_off)
    kernel = (1 - np.exp(-t_kernel / tau_on)) * np.exp(-t_kernel / tau_off)
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Deconvolve using Wiener filtering
    # Simple approach: use inverse filter with regularization
    Fc_fft = np.fft.fft(Fc)
    kernel_fft = np.fft.fft(kernel, len(Fc))
    
    # Wiener deconvolution with noise regularization
    noise_power = 0.01
    kernel_conj = np.conj(kernel_fft)
    deconv_filter = kernel_conj / (np.abs(kernel_fft)**2 + noise_power)
    
    Fc_deconv_fft = Fc_fft * deconv_filter
    Fc_deconv = np.real(np.fft.ifft(Fc_deconv_fft))
    
    return Fc_deconv


def find_contiguous_segments(binary_vector):
    """
    Find contiguous segments where binary_vector is True
    
    Parameters:
    -----------
    binary_vector : array-like
        Binary vector
    
    Returns:
    --------
    segments : list of arrays
        List of index arrays for each contiguous segment
    """
    binary_vector = np.asarray(binary_vector, dtype=bool)
    padded = np.concatenate(([False], binary_vector, [False]))
    transitions = np.diff(padded.astype(int))
    
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0]
    
    segments = [np.arange(start, end) for start, end in zip(start_indices, end_indices)]
    return segments


def build_design_matrix(speed, position, licking, reward, dt=0.1, maxlag=None, nbins=20):
    """
    Build design matrix for ridge regression
    
    Parameters:
    -----------
    speed : array-like
        Running speed
    position : array-like
        Position on track (normalized 0-1)
    licking : array-like
        Binary licking events
    reward : array-like
        Binary reward events
    dt : float
        Sampling interval in seconds
    maxlag : int, optional
        Maximum lag for event predictors (in samples)
    nbins : int
        Number of position bins
    
    Returns:
    --------
    X : numpy array
        Design matrix (n_samples x n_features)
    feature_names : list
        Names of features
    """
    if maxlag is None:
        maxlag = int(2 / dt)  # 2 seconds
    
    n_samples = len(speed)
    X = []
    feature_names = []
    
    # 1. Speed terms (linear and quadratic)
    X.append(speed.reshape(-1, 1))
    X.append((speed ** 2).reshape(-1, 1))
    feature_names.extend(['speed', 'speed^2'])
    
    # 2. Position bins
    if position is not None:
        edges = np.linspace(0, np.max(position), nbins + 1)
        for i in range(nbins):
            pos_bin = ((position >= edges[i]) & (position < edges[i + 1])).astype(float)
            X.append(pos_bin.reshape(-1, 1))
            feature_names.append(f'pos_bin_{i+1}')
    
    # 3. Event predictors with lags (interleaved: reward, licking)
    for lag in range(maxlag + 1):
        # Reward at this lag
        if lag == 0:
            reward_lagged = reward
        else:
            reward_lagged = np.concatenate([np.zeros(lag), reward[:-lag]])
        X.append(reward_lagged.reshape(-1, 1))
        feature_names.append(f'reward_lag_{lag}')
        
        # Licking at this lag
        if lag == 0:
            licking_lagged = licking
        else:
            licking_lagged = np.concatenate([np.zeros(lag), licking[:-lag]])
        X.append(licking_lagged.reshape(-1, 1))
        feature_names.append(f'licking_lag_{lag}')
    
    X = np.hstack(X)
    return X, feature_names


def ridge_cross_validation(X, Y, lambda_vals, k=10):
    """
    Perform k-fold cross-validation to find optimal lambda for ridge regression
    
    Parameters:
    -----------
    X : numpy array
        Design matrix (n_samples x n_features)
    Y : numpy array
        Target variable (n_samples,)
    lambda_vals : array-like
        Lambda values to test
    k : int
        Number of folds for cross-validation
    
    Returns:
    --------
    best_lambda : float
        Optimal lambda value
    mean_mse : numpy array
        Mean MSE for each lambda
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_vals = np.zeros((len(lambda_vals), k))
    
    for li, lambda_val in enumerate(lambda_vals):
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Fit ridge regression
            # Note: sklearn's Ridge uses alpha parameter (equivalent to lambda)
            model = Ridge(alpha=lambda_val, fit_intercept=True)
            model.fit(X_train, Y_train)
            
            # Predict and compute MSE
            Y_pred = model.predict(X_test)
            mse_vals[li, fold_idx] = np.mean((Y_test - Y_pred) ** 2)
    
    # Find best lambda
    mean_mse = np.mean(mse_vals, axis=1)
    best_idx = np.argmin(mean_mse)
    best_lambda = lambda_vals[best_idx]
    
    return best_lambda, mean_mse


def compute_partial_variance(X, Y, coefficients, feature_groups):
    """
    Compute variance explained by each group of features
    
    Parameters:
    -----------
    X : numpy array
        Design matrix
    Y : numpy array
        Target variable
    coefficients : numpy array
        Model coefficients (excluding intercept)
    feature_groups : dict
        Dictionary mapping group names to column indices
    
    Returns:
    --------
    var_explained : dict
        Variance explained by each group
    """
    Y_var = np.var(Y)
    var_explained = {}
    
    for group_name, indices in feature_groups.items():
        # Predict using only this group
        Y_pred_group = X[:, indices] @ coefficients[indices]
        var_group = np.var(Y_pred_group) / Y_var
        var_explained[group_name] = var_group
    
    return var_explained


def analyze_ach_single_session(mat_file_path, idx_start=None, idx_end=None, 
                                vel_threshold=0.5, include_position=True,
                                dt=0.1, make_plots=True):
    """
    Complete analysis pipeline for a single session
    
    Parameters:
    -----------
    mat_file_path : str
        Path to .mat file
    idx_start : int, optional
        Starting index for analysis
    idx_end : int, optional
        Ending index for analysis
    vel_threshold : float
        Velocity threshold (cm/s) for filtering
    include_position : bool
        Whether to include position predictors
    dt : float
        Sampling interval in seconds
    make_plots : bool
        Whether to generate plots
    
    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    print(f"Loading data from {mat_file_path}...")
    
    # Load .mat file
    data = loadmat(mat_file_path)
    
    # Extract variables (adjust these names based on your .mat file structure)
    speed = np.squeeze(data['v2'])
    licking = np.squeeze(data['licks2'])
    reward = np.squeeze(data['reward2'])
    position = np.squeeze(data['y2_norm']) if include_position else None
    Fc = np.squeeze(data['Fc_sp'])
    
    # Apply index range if specified
    if idx_start is not None and idx_end is not None:
        idx_range = slice(idx_start, idx_end)
        speed = speed[idx_range]
        licking = licking[idx_range]
        reward = reward[idx_range]
        if position is not None:
            position = position[idx_range]
        Fc = Fc[idx_range]
    
    print(f"Data shape: {len(speed)} samples")
    
    # Deconvolve fluorescence
    print("Deconvolving ACh signal...")
    sampling_rate = 1 / dt  # Hz
    tau_on = 0.06
    tau_off = 1.0
    Fc_deconv = deconvolve_grabACh_biExp(Fc, sampling_rate, tau_on, tau_off)
    
    # Apply velocity threshold
    above_thresh = speed > vel_threshold
    print(f"Samples above threshold: {np.sum(above_thresh)} ({100*np.mean(above_thresh):.1f}%)")
    
    speed_filt = speed[above_thresh]
    licking_filt = licking[above_thresh]
    reward_filt = reward[above_thresh]
    position_filt = position[above_thresh] if position is not None else None
    Y_filt = Fc_deconv[above_thresh]
    
    # Build design matrix
    print("Building design matrix...")
    maxlag = int(2 / dt)  # 2 seconds of lags
    nbins = 20
    X, feature_names = build_design_matrix(
        speed_filt, position_filt, licking_filt, reward_filt,
        dt=dt, maxlag=maxlag, nbins=nbins
    )
    
    print(f"Design matrix shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Cross-validation for lambda selection
    print("\nPerforming cross-validation for lambda selection...")
    lambda_vals = np.logspace(-2, 2, 10)
    best_lambda, mean_mse = ridge_cross_validation(X, Y_filt, lambda_vals, k=10)
    print(f"Best lambda: {best_lambda:.3f}")
    
    # Fit final model with best lambda
    print("\nFitting final ridge model...")
    model = Ridge(alpha=best_lambda, fit_intercept=True)
    model.fit(X, Y_filt)
    Y_pred_filt = model.predict(X)
    
    # Compute variance explained
    var_explained = 1 - np.var(Y_filt - Y_pred_filt) / np.var(Y_filt)
    print(f"Variance explained: {100 * var_explained:.2f}%")
    
    # Compute partial variance contributions
    feature_groups = {
        'Speed': [0, 1],  # speed and speed^2
    }
    
    if include_position:
        feature_groups['Position'] = list(range(2, 2 + nbins))
        event_start = 2 + nbins
    else:
        event_start = 2
    
    # Reward and licking indices (interleaved)
    reward_indices = list(range(event_start, event_start + 2 * (maxlag + 1), 2))
    licking_indices = list(range(event_start + 1, event_start + 2 * (maxlag + 1), 2))
    
    feature_groups['Reward'] = reward_indices
    feature_groups['Licking'] = licking_indices
    
    var_contributions = compute_partial_variance(X, Y_filt, model.coef_, feature_groups)
    
    print("\nVariance explained by each predictor group:")
    for group, var_val in var_contributions.items():
        print(f"  {group}: {100 * var_val:.2f}%")
    
    # Prepare results dictionary
    results = {
        'model': model,
        'best_lambda': best_lambda,
        'var_explained': var_explained,
        'var_contributions': var_contributions,
        'Y_full': Fc_deconv,
        'Y_filt': Y_filt,
        'Y_pred_filt': Y_pred_filt,
        'above_thresh': above_thresh,
        'X': X,
        'feature_names': feature_names,
        'feature_groups': feature_groups,
        'speed': speed,
        'speed_filt': speed_filt,
        'position': position,
        'vel_threshold': vel_threshold,
        'dt': dt,
        'maxlag': maxlag,
        'nbins': nbins,
        'include_position': include_position
    }
    
    # Generate plots
    if make_plots:
        plot_results(results)
    
    return results


def plot_results(results):
    """
    Create comprehensive visualization of analysis results
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_ach_single_session
    """
    fig = plt.figure(figsize=(16, 10))
    
    Y_full = results['Y_full']
    Y_filt = results['Y_filt']
    Y_pred_filt = results['Y_pred_filt']
    above_thresh = results['above_thresh']
    var_explained = results['var_explained']
    var_contributions = results['var_contributions']
    speed = results['speed']
    speed_filt = results['speed_filt']
    vel_threshold = results['vel_threshold']
    dt = results['dt']
    model = results['model']
    feature_groups = results['feature_groups']
    include_position = results['include_position']
    nbins = results['nbins']
    position = results['position']
    
    # 1. Actual vs Predicted scatter plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(Y_filt, Y_pred_filt, 'k.', markersize=2, alpha=0.5)
    lims = [min(Y_filt.min(), Y_pred_filt.min()), 
            max(Y_filt.max(), Y_pred_filt.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Unity line')
    ax1.set_xlabel('Actual ACh Signal', fontsize=12)
    ax1.set_ylabel('Predicted ACh Signal', fontsize=12)
    ax1.set_title(f'Ridge Model Fit (R² = {var_explained:.2f})\nVel > {vel_threshold:.1f} cm/s', 
                  fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series comparison
    ax2 = plt.subplot(2, 3, (2, 3))
    Y_pred_full = np.full(len(Y_full), np.nan)
    Y_pred_full[above_thresh] = Y_pred_filt
    t = np.arange(len(Y_full)) * dt
    
    ax2.plot(t, Y_full, 'k', linewidth=1, label='Actual', alpha=0.8)
    ax2.plot(t, Y_pred_full, 'r', linewidth=1, label='Predicted', alpha=0.8)
    
    # Shade regions below threshold
    segments = find_contiguous_segments(~above_thresh)
    for segment in segments:
        if len(segment) > 0:
            t_segment = t[segment]
            ax2.axvspan(t_segment[0], t_segment[-1], color='gray', alpha=0.3)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('ACh Signal', fontsize=12)
    ax2.set_title(f'Actual vs Predicted Time Series\n(Gray regions: speed < {vel_threshold:.1f} cm/s)', 
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Variance explained by predictors
    ax3 = plt.subplot(2, 3, 4)
    groups = list(var_contributions.keys())
    values = [var_contributions[g] for g in groups]
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E'][:len(groups)]
    bars = ax3.bar(groups, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Fraction of Variance Explained', fontsize=12)
    ax3.set_title('Contribution of Different Predictors', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Speed tuning curve
    ax4 = plt.subplot(2, 3, 5)
    max_speed = np.max(speed_filt)
    speed_range = np.linspace(vel_threshold, max_speed, 100)
    
    # Predict ACh for different speeds (with other predictors at mean)
    speed_pred = model.intercept_ + model.coef_[0] * speed_range + model.coef_[1] * speed_range**2
    
    ax4.plot(speed_range, speed_pred, linewidth=2, color='#0072BD')
    ax4.set_xlabel('Running Speed (cm/s)', fontsize=12)
    ax4.set_ylabel('ACh Response', fontsize=12)
    ax4.set_title(f'Speed Tuning (>{vel_threshold:.1f} cm/s)', fontsize=12)
    ax4.set_xlim([0, max_speed])
    ax4.grid(True, alpha=0.3)
    
    # 5. Position coefficients (if included)
    ax5 = plt.subplot(2, 3, 6)
    if include_position and position is not None:
        pos_coef = model.coef_[feature_groups['Position']]
        pos_bins = np.linspace(0, np.max(position), nbins)
        ax5.bar(pos_bins, pos_coef, width=pos_bins[1]-pos_bins[0], 
                color='#D95319', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Position', fontsize=12)
        ax5.set_ylabel('Coefficient', fontsize=12)
        ax5.set_title('Position Coefficients', fontsize=12)
    else:
        # Plot event kernels instead
        maxlag = results['maxlag']
        time_lags = np.arange(maxlag + 1) * dt
        
        reward_coef = model.coef_[feature_groups['Reward']]
        licking_coef = model.coef_[feature_groups['Licking']]
        
        ax5.plot(time_lags, reward_coef, 'o-', linewidth=2, 
                label='Reward', color='#EDB120', markersize=6)
        ax5.plot(time_lags, licking_coef, 's-', linewidth=2, 
                label='Licking', color='#7E2F8E', markersize=6)
        ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Time Lag (s)', fontsize=12)
        ax5.set_ylabel('Coefficient', fontsize=12)
        ax5.set_title('Temporal Kernels', fontsize=12)
        ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Ridge Regression Analysis Results\n({100*np.mean(above_thresh):.1f}% of data above {vel_threshold:.1f} cm/s)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Create separate figure for temporal kernels if position was included
    if include_position:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        maxlag = results['maxlag']
        time_lags = np.arange(maxlag + 1) * dt
        
        reward_coef = model.coef_[feature_groups['Reward']]
        licking_coef = model.coef_[feature_groups['Licking']]
        
        ax.plot(time_lags, reward_coef, 'o-', linewidth=2, 
                label='Reward', color='#EDB120', markersize=8)
        ax.plot(time_lags, licking_coef, 's-', linewidth=2, 
                label='Licking', color='#7E2F8E', markersize=8)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Time Lag (s)', fontsize=14)
        ax.set_ylabel('Coefficient', fontsize=14)
        ax.set_title('Temporal Kernels for Events', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def analyze_multiple_sessions(session_files, session_idx_ranges=None, 
                              vel_threshold=0.5, include_position=False):
    """
    Analyze multiple sessions and compare results
    
    Parameters:
    -----------
    session_files : list
        List of paths to .mat files
    session_idx_ranges : list of tuples, optional
        List of (start, end) index ranges for each session
    vel_threshold : float
        Velocity threshold for filtering
    include_position : bool
        Whether to include position predictors
    
    Returns:
    --------
    all_results : list
        List of results dictionaries for each session
    """
    num_sessions = len(session_files)
    all_results = []
    best_lambdas = []
    var_explained_all = []
    var_contributions_all = {
        'Speed': [],
        'Reward': [],
        'Licking': []
    }
    
    if include_position:
        var_contributions_all['Position'] = []
    
    for s, file_path in enumerate(session_files):
        print(f"\n{'='*60}")
        print(f"Processing Session {s+1}/{num_sessions}")
        print(f"{'='*60}")
        
        if session_idx_ranges is not None:
            idx_start, idx_end = session_idx_ranges[s]
        else:
            idx_start, idx_end = None, None
        
        results = analyze_ach_single_session(
            file_path, idx_start, idx_end,
            vel_threshold=vel_threshold,
            include_position=include_position,
            make_plots=False
        )
        
        all_results.append(results)
        best_lambdas.append(results['best_lambda'])
        var_explained_all.append(results['var_explained'])
        
        for key in var_contributions_all.keys():
            var_contributions_all[key].append(results['var_contributions'][key])
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Across Sessions")
    print(f"{'='*60}")
    print(f"Lambda statistics:")
    print(f"  Mean: {np.mean(best_lambdas):.4f}")
    print(f"  Median: {np.median(best_lambdas):.4f}")
    print(f"  Min: {np.min(best_lambdas):.4f}")
    print(f"  Max: {np.max(best_lambdas):.4f}")
    print(f"  Std: {np.std(best_lambdas):.4f}")
    
    # Plot summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Lambda values
    ax1 = axes[0, 0]
    ax1.bar(range(1, num_sessions+1), best_lambdas, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Session', fontsize=12)
    ax1.set_ylabel('λ', fontsize=12)
    ax1.set_title('Optimal λ per Session', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Variance explained
    ax2 = axes[0, 1]
    ax2.bar(range(1, num_sessions+1), var_explained_all, color='coral', 
            edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Session', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Variance Explained per Session', fontsize=12)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Variance contributions
    ax3 = axes[1, 0]
    x = np.arange(num_sessions)
    width = 0.2
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    
    for i, (key, values) in enumerate(var_contributions_all.items()):
        offset = width * (i - len(var_contributions_all)/2 + 0.5)
        ax3.bar(x + offset + 1, values, width, label=key, 
                color=colors[i], alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('Session', fontsize=12)
    ax3.set_ylabel('Variance Explained', fontsize=12)
    ax3.set_title('Variance Contributions by Predictor Group', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Average contributions
    ax4 = axes[1, 1]
    avg_contributions = {key: np.mean(values) for key, values in var_contributions_all.items()}
    ax4.bar(avg_contributions.keys(), avg_contributions.values(), 
            color=colors[:len(avg_contributions)], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Mean Variance Explained', fontsize=12)
    ax4.set_title('Average Contributions Across Sessions', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (key, val) in enumerate(avg_contributions.items()):
        ax4.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Multi-Session Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return all_results


# Example usage for Jupyter Notebook
if __name__ == "__main__":
    """
    Example usage in Jupyter Notebook:
    
    # Single session analysis
    results = analyze_ach_single_session(
        mat_file_path='path/to/your/data.mat',
        idx_start=0,
        idx_end=10000,
        vel_threshold=0.5,
        include_position=True,
        make_plots=True
    )
    
    # Access results
    print(f"Best lambda: {results['best_lambda']}")
    print(f"Variance explained: {results['var_explained']:.2%}")
    print("Variance contributions:", results['var_contributions'])
    
    # Multiple session analysis
    session_files = [
        'path/to/session1.mat',
        'path/to/session2.mat',
        'path/to/session3.mat'
    ]
    
    all_results = analyze_multiple_sessions(
        session_files,
        vel_threshold=0.5,
        include_position=False
    )
    """
    print("ACh GLM Analysis Module Loaded")
    print("Use analyze_ach_single_session() for single session analysis")
    print("Use analyze_multiple_sessions() for multi-session analysis")
