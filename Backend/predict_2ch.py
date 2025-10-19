"""
2-Channel EMG Clench Detection - Inference Module
FIXED VERSION: Better dummy data generation
Works with existing 18-D model (3 IMFs)
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import warnings
from emd_utils import EMD, compute_cct_matrix_fast

warnings.filterwarnings('ignore', category=UserWarning)

# ========== MODEL DEFINITION ==========
class ClenchTransformer2Ch(nn.Module):
    """
    Optimized for 2-channel (18-D input)
    """
    def __init__(self, input_dim=18, d_model=32, num_heads=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        
        self.out = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.dropout1(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.out(x)

# ========== LOAD MODEL & SCALER ==========
try:
    model = ClenchTransformer2Ch()
    model.load_state_dict(torch.load("model/final_2ch_clench_transformer.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load("model/scaler_2ch.pkl")
    print("✅ 2-Channel model loaded successfully")
except Exception as e:
    print(f"⚠️  Model loading error: {e}")
    print("⚠️  Using untrained model (predictions will be random)")
    model = ClenchTransformer2Ch()
    model.eval()
    scaler = None

# ========== FEATURE EXTRACTION (18-D) ==========
_emd_instance = EMD(max_imfs=3)

def extract_features_2ch(window):
    """
    Extract EMD-CCT features from 2-channel window (2, 300) -> 18-D vector
    
    Args:
        window: (2, 300) numpy array [CH0, CH1]
    
    Returns:
        18-D feature vector (always valid, never None)
    """
    feats = []
    
    for ch in range(2):
        imfs = _emd_instance.emd(window[ch])
        
        # Handle edge cases (same as training)
        if imfs.shape[0] == 0:
            imfs = np.zeros((3, window.shape[1]))
        elif imfs.shape[0] < 3:
            padding = np.zeros((3 - imfs.shape[0], window.shape[1]))
            imfs = np.vstack([imfs, padding])
        
        imfs = imfs[:3]
        
        cct = compute_cct_matrix_fast(imfs)
        feats.extend(cct.flatten())  # 9 per channel
    
    return np.array(feats, dtype=np.float32)


def scale_features_2ch(x):
    """Scale 18-D features"""
    if scaler is None:
        # No scaler - return normalized features
        return (x - x.mean()) / (x.std() + 1e-8)
    
    x = x.reshape(1, -1)
    if x.shape[1] != 18:
        raise ValueError(f"Expected 18 features, got {x.shape[1]}")
    return scaler.transform(x).flatten()


# ========== IMPROVED REALISTIC 2-CHANNEL EMG ==========
def generate_realistic_2ch_emg(clench=False, snr_db=18):
    """
    FIXED: Generate realistic 2-channel EMG that matches training data
    
    Key improvements:
    - More realistic muscle activation patterns
    - Proper frequency content (15-40 Hz muscle activity)
    - Realistic amplitude ranges
    - Better noise characteristics
    
    CH0: Flexor Digitorum (finger flexion) - ACTIVE when clenching
    CH1: Extensor Digitorum (finger extension) - ACTIVE when open
    
    Returns: (2, 300) array
    """
    dummy = np.zeros((2, 300))
    t = np.arange(300) / 100  # 100 Hz sampling, 3 seconds
    
    if clench:
        # ========== CLENCH STATE ==========
        
        # Channel 0 (Flexor) - STRONG ACTIVATION
        # Realistic muscle activation envelope
        activation_time = np.zeros(300)
        for i, time in enumerate(t):
            if time < 0.3:
                # Rest before clench
                activation_time[i] = 0.15
            elif time < 0.5:
                # Ramp up (200ms)
                activation_time[i] = 0.15 + 2.35 * (time - 0.3) / 0.2
            elif time < 2.5:
                # Sustained contraction with natural variation
                activation_time[i] = 2.5 + 0.4 * np.sin(2 * np.pi * 1.5 * time)
            else:
                # Relaxation phase
                activation_time[i] = 2.5 * np.exp(-4 * (time - 2.5))
        
        # Muscle fiber firing (multiple motor units, 20-40 Hz)
        muscle_activity = np.zeros(300)
        num_motor_units = np.random.randint(4, 7)
        for _ in range(num_motor_units):
            freq = np.random.uniform(22, 38)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.7, 1.3)
            muscle_activity += amplitude * np.sin(2 * np.pi * freq * t + phase) / num_motor_units
        
        signal_flex = activation_time * muscle_activity
        
        # Add realistic action potential bursts
        num_bursts = np.random.randint(10, 18)
        burst_times = np.random.choice(range(30, 270), size=num_bursts, replace=False)
        for burst_time in burst_times:
            if burst_time < 295:
                # Biphasic spike (realistic shape)
                spike_shape = np.array([0, 1.8, -1.4, 0.5, 0])
                spike_amplitude = np.random.uniform(0.7, 1.5)
                signal_flex[burst_time:burst_time+5] += spike_shape * spike_amplitude
        
        dummy[0] = signal_flex
        
        # Channel 1 (Extensor) - LOW ACTIVITY (antagonist muscle)
        # Small baseline activity (co-contraction)
        baseline_noise = np.random.randn(300) * 0.25
        small_activity = 0.15 * np.sin(2 * np.pi * 18 * t)
        dummy[1] = baseline_noise + small_activity
        
    else:
        # ========== OPEN/REST STATE ==========
        
        # Channel 0 (Flexor) - MINIMAL ACTIVITY
        baseline = np.random.randn(300) * 0.18
        
        # Occasional small twitches (muscle tone)
        if np.random.rand() > 0.85:
            twitch_start = np.random.randint(50, 240)
            twitch_duration = np.random.randint(15, 35)
            if twitch_start + twitch_duration < 300:
                twitch_envelope = np.sin(np.linspace(0, np.pi, twitch_duration))
                twitch_signal = twitch_envelope * np.random.uniform(0.4, 0.7)
                baseline[twitch_start:twitch_start+twitch_duration] += twitch_signal
        
        dummy[0] = baseline
        
        # Channel 1 (Extensor) - MODERATE ACTIVITY (keeps hand open)
        # Sustained moderate contraction
        activation = np.ones(300) * 1.0
        # Add variation (tremor, fatigue)
        activation += 0.2 * np.sin(2 * np.pi * 2.5 * t)
        
        # Motor unit activity (lower frequency for postural control)
        carrier = np.zeros(300)
        for freq in np.random.uniform(15, 28, 4):
            phase = np.random.uniform(0, 2 * np.pi)
            carrier += np.sin(2 * np.pi * freq * t + phase) / 4
        
        signal_ext = activation * carrier * 0.45
        
        # Add some baseline noise
        signal_ext += np.random.randn(300) * 0.22
        
        dummy[1] = signal_ext
    
    # ========== ADD REALISTIC NOISE TO BOTH CHANNELS ==========
    for ch in range(2):
        signal_power = np.var(dummy[ch])
        
        # White Gaussian noise (thermal, electronic)
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            white_noise = np.random.randn(300) * np.sqrt(noise_power)
        else:
            white_noise = np.random.randn(300) * 0.05
        
        # 50 Hz powerline interference (common in EMG)
        powerline_amplitude = np.random.uniform(0.015, 0.025)
        powerline_phase = np.random.uniform(0, 2 * np.pi)
        powerline = powerline_amplitude * np.sin(2 * np.pi * 50 * t + powerline_phase)
        
        # Motion artifacts (low frequency, 0.5-2 Hz)
        motion_freq = np.random.uniform(0.4, 1.8)
        motion_amplitude = np.random.uniform(0.08, 0.15)
        motion = motion_amplitude * np.sin(2 * np.pi * motion_freq * t) * np.random.rand()
        
        # High frequency noise (electrode-skin interface)
        hf_noise = np.random.randn(300) * 0.025
        
        # Baseline drift (very low frequency)
        drift = np.random.uniform(-0.05, 0.05) * np.linspace(0, 1, 300)
        
        # Combine all noise sources
        dummy[ch] += white_noise + powerline + motion + hf_noise + drift
    
    return dummy


# ========== PREDICTION ==========
_prediction_counter = 0
_recent_predictions = []  # For smoothing

def predict_clench_2ch():
    """
    Predict clench from 2-channel EMG window
    
    FIXED: Uses realistic dummy data and prediction smoothing
    
    Returns: 
        [clench, False, False, False, False]
    """
    global _prediction_counter, _recent_predictions
    _prediction_counter += 1
    
    # Generate realistic dummy data
    # Simulate: 70% open, 30% clench (realistic usage pattern)
    is_clench_phase = (_prediction_counter % 10) < 3
    dummy = generate_realistic_2ch_emg(clench=is_clench_phase, snr_db=18)
    
    # Extract features (18-D)
    feat = extract_features_2ch(dummy)
    
    # Scale and predict
    feat = scale_features_2ch(feat)
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 18)
    
    with torch.no_grad():
        prob = model(feat_tensor).squeeze().item()
        
        # Use threshold
        clench = prob > 0.5
        confidence = abs(prob - 0.5) * 2  # 0 to 1 scale
        
        # Smoothing: Track recent predictions (reduces jitter)
        _recent_predictions.append(prob)
        if len(_recent_predictions) > 5:
            _recent_predictions.pop(0)
        
        avg_prob = np.mean(_recent_predictions)
        stable_clench = avg_prob > 0.5
        
        # Visual indicator
        status_emoji = "✊" if stable_clench else "✋"
        
        # Print detailed info
        print(f"{status_emoji} [2-CH] Pred #{_prediction_counter:3d}: "
              f"prob={prob:.3f}, avg={avg_prob:.3f}, "
              f"clench={stable_clench}, conf={confidence:.2f}, "
              f"actual={'CLENCH' if is_clench_phase else 'OPEN  '}")
    
    return [stable_clench, False, False, False, False]


# ========== DIAGNOSTIC TEST ==========
def test_model_with_extreme_inputs():
    """
    Test model with extreme inputs to verify it's not stuck
    """
    print("\n🔬 DIAGNOSTIC TEST: Testing model with extreme inputs\n")
    
    # Test 1: All zeros (should predict open)
    print("Test 1: All zeros input")
    feat_zeros = np.zeros(18, dtype=np.float32)
    feat_zeros_scaled = scale_features_2ch(feat_zeros)
    feat_tensor = torch.tensor(feat_zeros_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prob_zeros = model(feat_tensor).squeeze().item()
        print(f"   Input: all zeros → Probability: {prob_zeros:.3f} → Prediction: {'CLENCH' if prob_zeros > 0.5 else 'OPEN'}")
    
    # Test 2: All ones (should predict clench)
    print("\nTest 2: All ones input")
    feat_ones = np.ones(18, dtype=np.float32)
    feat_ones_scaled = scale_features_2ch(feat_ones)
    feat_tensor = torch.tensor(feat_ones_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prob_ones = model(feat_tensor).squeeze().item()
        print(f"   Input: all ones → Probability: {prob_ones:.3f} → Prediction: {'CLENCH' if prob_ones > 0.5 else 'OPEN'}")
    
    # Test 3: Random values
    print("\nTest 3: Random values")
    feat_random = np.random.randn(18).astype(np.float32)
    feat_random_scaled = scale_features_2ch(feat_random)
    feat_tensor = torch.tensor(feat_random_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prob_random = model(feat_tensor).squeeze().item()
        print(f"   Input: random → Probability: {prob_random:.3f} → Prediction: {'CLENCH' if prob_random > 0.5 else 'OPEN'}")
    
    # Verdict
    print("\n" + "="*70)
    if abs(prob_zeros - prob_ones) < 0.01:
        print("⚠️  WARNING: Model outputs are nearly identical!")
        print("   This suggests the model is NOT properly trained.")
        print("   Recommendation: Retrain the model with proper data.")
    else:
        print("✅ Model responds to different inputs (working correctly)")
        print("   Issue is likely with feature extraction or dummy data.")
    print("="*70 + "\n")


# ========== TESTING ==========
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing 2-Channel EMG Clench Detection (Dummy Data)")
    print("="*70)
    print(f"Model loaded: {scaler is not None}")
    print("="*70 + "\n")
    
    # Run diagnostic first
    test_model_with_extreme_inputs()
    
    print("\nRunning 30 predictions (you should see both ✊ and ✋):\n")
    
    for i in range(30):
        result = predict_clench_2ch()
        
        # Small delay to simulate real-time
        import time
        time.sleep(0.1)
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print(f"Total predictions: {_prediction_counter}")
    print("="*70)
    
    # Summary
    print("\n📊 Expected behavior:")
    print("   - Predictions 1-3: CLENCH (✊)")
    print("   - Predictions 4-10: OPEN (✋)")
    print("   - Predictions 11-13: CLENCH (✊)")
    print("   - Pattern repeats...")