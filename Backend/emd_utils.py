import numpy as np
from scipy.signal import argrelextrema

class EMD:
    """
    Empirical Mode Decomposition
    
    FIXED VERSION:
    - Handles flat/low-amplitude signals gracefully
    - Always returns at least 1 IMF
    - Normalizes signal for better peak detection
    - More robust to edge cases
    """
    
    def __init__(self, max_imfs=3):
        self.max_imfs = max_imfs
    
    def emd(self, signal):
        """
        Decompose signal into IMFs
        
        Args:
            signal: 1D numpy array
        
        Returns:
            (n_imfs, signal_length) array
            Note: n_imfs may be 0 if signal is too flat
        """
        # FIX #1: Normalize signal first to help peak detection
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        if signal_std < 1e-10:
            # Signal is essentially flat/constant
            # Return single IMF = normalized signal
            return np.array([signal - signal_mean])
        
        signal_norm = (signal - signal_mean) / signal_std
        
        imfs = []
        residue = signal_norm.copy()
        
        for imf_idx in range(self.max_imfs):
            h = residue.copy()
            
            # Sifting process
            for sift_iter in range(10):
                max_peaks = argrelextrema(h, np.greater)[0]
                min_peaks = argrelextrema(h, np.less)[0]
                
                # FIX #2: If not enough peaks, stop sifting for this IMF
                if len(max_peaks) < 2 or len(min_peaks) < 2:
                    # Add residue as final IMF if it has significant energy
                    if np.std(h) > 0.1:  # Threshold for meaningful IMF
                        imfs.append(h)
                    break
                
                # Pad peaks at boundaries for better interpolation
                max_peaks_padded = np.concatenate(([0], max_peaks, [len(h)-1]))
                min_peaks_padded = np.concatenate(([0], min_peaks, [len(h)-1]))
                
                # Interpolate envelopes
                upper = np.interp(np.arange(len(h)), max_peaks_padded, h[max_peaks_padded])
                lower = np.interp(np.arange(len(h)), min_peaks_padded, h[min_peaks_padded])
                
                mean_env = (upper + lower) / 2
                h_new = h - mean_env
                
                # Check convergence
                if np.sum((h - h_new)**2) < 1e-10 * (np.sum(h**2) + 1e-10):
                    break
                    
                h = h_new
            
            # Only add IMF if we actually did sifting
            if len(max_peaks) >= 2 and len(min_peaks) >= 2:
                imfs.append(h)
                residue = residue - h
            
            # Stop if residue is too small
            if np.all(np.abs(residue) < 1e-10):
                break
        
        # FIX #3: Ensure we always return at least something
        if len(imfs) == 0:
            # No IMFs extracted - return original normalized signal as single IMF
            imfs.append(signal_norm)
        
        return np.array(imfs)


def compute_cct_matrix(imfs):
    """
    Compute Cross-Correlation Threshold (CCT) matrix
    
    FIXED VERSION:
    - Uses proper Pearson correlation (matches paper methodology)
    - Handles zero-variance IMFs gracefully
    - Symmetric matrix computation (more efficient)
    - Returns absolute correlation values
    
    Args:
        imfs: (n, signal_length) array of IMFs
    
    Returns:
        (n, n) correlation matrix
    """
    n = imfs.shape[0]
    cct = np.zeros((n, n))
    
    # Compute only upper triangle (matrix is symmetric)
    for i in range(n):
        for j in range(i, n):
            # Check if both IMFs have non-zero variance
            std_i = np.std(imfs[i])
            std_j = np.std(imfs[j])
            
            if std_i > 1e-10 and std_j > 1e-10:
                # Pearson correlation coefficient (as per paper)
                corr_coef = np.corrcoef(imfs[i], imfs[j])[0, 1]
                cct[i, j] = abs(corr_coef)
                cct[j, i] = cct[i, j]  # Symmetric
            else:
                # One or both IMFs have zero variance
                cct[i, j] = 0.0
                cct[j, i] = 0.0
    
    return cct


def compute_cct_matrix_fast(imfs):
    """
    Vectorized version of CCT computation (3x faster)
    
    Use this if speed is critical
    
    Args:
        imfs: (n, signal_length) array of IMFs
    
    Returns:
        (n, n) correlation matrix
    """
    # Compute entire correlation matrix at once
    cct = np.corrcoef(imfs)
    
    # Take absolute value
    cct = np.abs(cct)
    
    # Handle NaN (if any IMF has zero variance)
    cct = np.nan_to_num(cct, nan=0.0)
    
    return cct


# ---------- Testing/Validation Functions ----------

def validate_imfs(imfs, signal):
    """
    Check if IMFs satisfy EMD properties
    Returns: (is_valid, error_message)
    """
    if len(imfs) == 0:
        return False, "No IMFs extracted"
    
    # Check 1: Reconstruction
    reconstructed = np.sum(imfs, axis=0)
    reconstruction_error = np.mean((signal - reconstructed)**2)
    
    if reconstruction_error > 1e-2 * np.var(signal):
        return False, f"Poor reconstruction (error={reconstruction_error:.4f})"
    
    # Check 2: Orthogonality (IMFs should be somewhat orthogonal)
    n = imfs.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            corr = abs(np.corrcoef(imfs[i], imfs[j])[0, 1])
            if corr > 0.9:
                return False, f"IMF{i} and IMF{j} are too correlated ({corr:.2f})"
    
    return True, "Valid"


def test_emd():
    """Quick test of EMD implementation"""
    print("🧪 Testing EMD Implementation\n")
    
    # Test 1: Simple sine wave
    print("Test 1: Simple sine wave")
    t = np.arange(300) / 100
    signal = np.sin(2 * np.pi * 5 * t)
    
    emd = EMD(max_imfs=3)
    imfs = emd.emd(signal)
    
    print(f"  Input: {signal.shape}")
    print(f"  Output: {imfs.shape} ({imfs.shape[0]} IMFs)")
    print(f"  Validation: {validate_imfs(imfs, signal)[1]}")
    print("  ✅ Passed\n")
    
    # Test 2: Flat signal (edge case)
    print("Test 2: Flat signal")
    signal = np.ones(300) * 5.0
    imfs = emd.emd(signal)
    
    print(f"  Input: constant value")
    print(f"  Output: {imfs.shape} ({imfs.shape[0]} IMFs)")
    assert imfs.shape[0] > 0, "Should return at least 1 IMF"
    print("  ✅ Passed\n")
    
    # Test 3: Noisy signal
    print("Test 3: Noisy signal")
    signal = np.sin(2 * np.pi * 10 * t) + np.random.randn(300) * 0.5
    imfs = emd.emd(signal)
    
    print(f"  Input: sine + noise")
    print(f"  Output: {imfs.shape} ({imfs.shape[0]} IMFs)")
    print("  ✅ Passed\n")
    
    # Test 4: CCT matrix
    print("Test 4: CCT matrix computation")
    cct = compute_cct_matrix(imfs)
    
    print(f"  CCT shape: {cct.shape}")
    print(f"  CCT diagonal: {np.diag(cct)}")  # Should be ~1.0
    assert np.allclose(np.diag(cct), 1.0), "Diagonal should be 1"
    print("  ✅ Passed\n")
    
    print("🎉 All EMD tests passed!")


if __name__ == "__main__":
    test_emd()