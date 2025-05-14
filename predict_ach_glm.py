import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns

def deconvolve_grabACh_biExp(Fc, sampling_rate, tau_on, tau_off):
    """
    Deconvolve ACh signal using biexponential kernel
    """
    # Create time vector
    t = np.arange(0, 10, 1/sampling_rate)
    
    # Create biexponential kernel
    kernel = np.exp(-t/tau_off) - np.exp(-t/tau_on)
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Deconvolve using scipy's deconvolve
    deconv, _ = signal.deconvolve(Fc, kernel)
    
    return deconv

def build_design_matrix(speed, position, reward, licking, dt=0.1, nbins=20, maxlag=None):
    """
    Build design matrix with predictors
    """
    if maxlag is None:
        maxlag = int(2/dt)  # 2-second window
    
    X = []
    
    # Speed terms (linear and quadratic)
    X.append(speed)
    X.append(speed**2)
    
    # Position terms (using basis functions)
    pos_basis = np.zeros((len(position), nbins))
    edges = np.linspace(0, np.max(position), nbins+1)
    for i in range(nbins):
        pos_basis[:, i] = (position >= edges[i]) & (position < edges[i+1])
    X.append(pos_basis)
    
    # Event terms with different time lags
    for lag in range(maxlag + 1):
        # Reward
        reward_lagged = np.concatenate([np.zeros(lag), reward[:-lag] if lag > 0 else reward])
        X.append(reward_lagged)
        
        # Licking
        licking_lagged = np.concatenate([np.zeros(lag), licking[:-lag] if lag > 0 else licking])
        X.append(licking_lagged)
    
    return np.column_stack(X)

def find_optimal_lambda(X, y, k=10, lambda_range=np.logspace(-2, 2, 10)):
    """
    Find optimal lambda using k-fold cross-validation
    """
    kf = KFold(n_splits=k, shuffle=True)
    mse_vals = np.zeros((len(lambda_range), k))
    
    for i, lambda_val in enumerate(lambda_range):
        for j, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit ridge regression
            model = Ridge(alpha=lambda_val)
            model.fit(X_train, y_train)
            
            # Predict and compute MSE
            y_pred = model.predict(X_test)
            mse_vals[i, j] = np.mean((y_test - y_pred)**2)
    
    # Find best lambda
    mean_mse = np.mean(mse_vals, axis=1)
    best_idx = np.argmin(mean_mse)
    return lambda_range[best_idx]

def analyze_ach_signal(data_path, idx_range=None):
    """
    Main function to analyze ACh signal
    """
    # Load data
    data = loadmat(data_path)
    
    # Extract variables
    speed = data['v2'].flatten()
    position = data['y2_norm'].flatten()
    reward = data['reward2'].flatten()
    licking = data['licks2'].flatten()
    Fc = data['Fc_sp'].flatten()
    
    if idx_range is not None:
        speed = speed[idx_range]
        position = position[idx_range]
        reward = reward[idx_range]
        licking = licking[idx_range]
        Fc = Fc[idx_range]
    
    # Deconvolve ACh signal
    sampling_rate = 10  # Hz
    tau_on = 0.06
    tau_off = 1
    Fc_deconv = deconvolve_grabACh_biExp(Fc, sampling_rate, tau_on, tau_off)
    
    # Apply velocity threshold
    vel_threshold = 0.5
    mask = speed >= vel_threshold
    
    # Filter data
    speed_filt = speed[mask]
    position_filt = position[mask]
    reward_filt = reward[mask]
    licking_filt = licking[mask]
    y_filt = Fc_deconv[mask]
    
    # Build design matrix
    X = build_design_matrix(speed_filt, position_filt, reward_filt, licking_filt)
    
    # Find optimal lambda
    best_lambda = find_optimal_lambda(X, y_filt)
    print(f"Best lambda: {best_lambda:.3f}")
    
    # Fit final model
    model = Ridge(alpha=best_lambda)
    model.fit(X, y_filt)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate variance explained
    var_explained = 1 - np.var(y_filt - y_pred) / np.var(y_filt)
    print(f"Variance explained: {var_explained*100:.2f}%")
    
    # Calculate partial variance contributions
    # Speed terms (first 2 columns)
    var_speed = np.var(X[:, :2] @ model.coef_[:2]) / np.var(y_filt)
    
    # Position terms (next 20 columns)
    var_pos = np.var(X[:, 2:22] @ model.coef_[2:22]) / np.var(y_filt)
    
    # Event terms
    maxlag = int(2/0.1)  # 2-second window
    reward_idx = np.arange(22, 22 + 2*(maxlag+1), 2)
    licking_idx = np.arange(23, 23 + 2*(maxlag+1), 2)
    
    var_reward = np.var(X[:, reward_idx] @ model.coef_[reward_idx]) / np.var(y_filt)
    var_licking = np.var(X[:, licking_idx] @ model.coef_[licking_idx]) / np.var(y_filt)
    
    print(f"Variance explained by speed: {var_speed*100:.2f}%")
    print(f"Variance explained by position: {var_pos*100:.2f}%")
    print(f"Variance explained by reward: {var_reward*100:.2f}%")
    print(f"Variance explained by licking: {var_licking*100:.2f}%")
    
    # Plotting
    plot_results(y_filt, y_pred, var_explained, var_speed, var_pos, var_reward, var_licking)
    
    return model, var_explained, (var_speed, var_pos, var_reward, var_licking)

def plot_results(y_true, y_pred, var_explained, var_speed, var_pos, var_reward, var_licking):
    """
    Create visualization plots
    """
    plt.figure(figsize=(15, 10))
    
    # Actual vs Predicted scatter plot
    plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=1)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual ACh Signal')
    plt.ylabel('Predicted ACh Signal')
    plt.title(f'Ridge Model Fit (R² = {var_explained:.2f})')
    
    # Time series comparison
    plt.subplot(2, 3, [2, 3])
    plt.plot(y_true, 'k-', label='Actual')
    plt.plot(y_pred, 'r-', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('ACh Signal')
    plt.title('Actual vs Predicted Time Series')
    plt.legend()
    
    # Variance explained by predictors
    plt.subplot(2, 3, 4)
    var_types = [var_speed, var_pos, var_reward, var_licking]
    plt.bar(['Speed', 'Position', 'Reward', 'Licking'], var_types)
    plt.ylabel('Fraction of Variance Explained')
    plt.title('Contribution of Different Predictors')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace with your data path
    data_path = "path_to_your_data.mat"
    idx_range = slice(1001, 4000)  # Example index range
    
    model, var_explained, var_contributions = analyze_ach_signal(data_path, idx_range)
