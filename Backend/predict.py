"""
FIXED 10-Channel EMG Clench Detection
Better dummy data generation for realistic testing
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import warnings
from emd_utils import EMD, compute_cct_matrix

warnings.filterwarnings('ignore', category=UserWarning)

# ========== MODEL DEFINITION ==========
class ClenchTransformer(nn.Module):
    def __init__(self, input_dim=90, d_model=64, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.out = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.out(x)

# ========== LOAD MODEL ==========
try:
    model = ClenchTransformer()
    model.load_state_dict(torch.load("model/final_clench_transformer.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load("model/scaler.pkl")
    print("✅ 10-Channel model loaded successfully")
except Exception as e:
    print(f"⚠️  Model loading error: {e}")
    print("⚠️  Using untrained model (predictions will be random)")
    model = ClenchTransformer()
    model.eval()
    scaler = None

# ========== CACHE EMD INSTANCE ==========
_emd_instance = EMD(max_imfs=3)

# ========== FEATURE EXTRACTION (90-D) ==========
def extract_features(window):  
    """
    Extract EMD-CCT features from window (10, 300) -> 90-D vector
    
    Args:
        window: (10, 300) numpy array
    
    Returns:
        90-D feature vector (always valid)
    """
    feats = []
    
    for ch in range(10):
        imfs = _emd_instance.emd(window[ch])
        
        # Handle edge cases
        if imfs.shape[0] == 0:
            imfs = np.zeros((3, window.shape[1]))
        elif imfs.shape[0] < 3:
            padding = np.zeros((3 - imfs.shape[0], window.shape[1]))
            imfs = np.vstack([imfs, padding])
        
        # Take first 3 IMFs
        imfs = imfs[:3]
        
        # Compute CCT matrix
        cct = compute_cct_matrix(imfs)
        feats.extend(cct.flatten())  # 9 per channel
    
    return np.array(feats, dtype=np.float32)

# ========== SAFE SCALING ==========
def scale_features(x):
    """Scale features using pre-trained scaler"""
    if scaler is None:
        return (x - x.mean()) / (x.std() + 1e-8)
    
    x = x.reshape(1, -1)
    if x.shape[1] != 90:
        raise ValueError(f"Expected 90 features, got {x.shape[1]}")
    return scaler.transform(x).flatten()

# ========== IMPROVED REALISTIC DUMMY EMG (10-CHANNEL) ==========
def generate_realistic_emg(clench=False, snr_db=18):
    """
    FIXED: Generate realistic 10-channel EMG
    
    Simulates NinaPro DB1 electrode placement on forearm
    Channels represent different forearm muscles
    
    Args:
        clench: If True, generate clenched pattern
        snr_db: Signal-to-noise ratio
    
    Returns:
        (10, 300) array
    """
    dummy = np.zeros((10, 300))
    t = np.arange(300) / 100  # 100 Hz, 3 seconds
    
    # Channel grouping (simulating muscle groups)
    # Channels 0-4: Flexors (palm side)
    # Channels 5-9: Extensors (back side)
    
    if clench:
        # ===== CLENCH STATE =====
        
        # Flexor channels (0-4) - HIGH ACTIVITY
        for ch in range(5):
            # Activation envelope with variation per channel
            activation = np.zeros(300)
            delay = np.random.uniform(0, 0.1)  # Slight recruitment delay
            
            for i, time in enumerate(t):
                if time < 0.3 + delay:
                    activation[i] = 0.12
                elif time < 0.5 + delay:
                    activation[i] = 0.12 + 2.38 * (time - 0.3 - delay) / 0.2
                elif time < 2.5:
                    activation[i] = 2.5 + 0.35 * np.sin(2 * np.pi * 1.8 * time)
                else:
                    activation[i] = 2.5 * np.exp(-4 * (time - 2.5))
            
            # Motor unit activity
            carrier = np.zeros(300)
            num_units = np.random.randint(3, 6)
            for _ in range(num_units):
                freq = np.random.uniform(20, 38)
                phase = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.6, 1.4)
                carrier += amp * np.sin(2 * np.pi * freq * t + phase) / num_units
            
            signal = activation * carrier
            
            # Action potential bursts
            num_bursts = np.random.randint(8, 15)
            burst_times = np.random.choice(range(30, 270), size=num_bursts, replace=False)
            for bt in burst_times:
                if bt < 295:
                    spike = np.array([0, 1.6, -1.3, 0.4, 0]) * np.random.uniform(0.7, 1.3)
                    signal[bt:bt+5] += spike
            
            dummy[ch] = signal
        
        # Extensor channels (5-9) - LOW ACTIVITY (antagonists)
        for ch in range(5, 10):
            baseline = np.random.randn(300) * 0.28
            small_activity = 0.18 * np.sin(2 * np.pi * np.random.uniform(15, 22) * t)
            dummy[ch] = baseline + small_activity
            
    else:
        # ===== OPEN/REST STATE =====
        
        # Flexor channels (0-4) - LOW ACTIVITY
        for ch in range(5):
            baseline = np.random.randn(300) * 0.2
            
            # Random twitches
            if np.random.rand() > 0.88:
                twitch_start = np.random.randint(50, 240)
                twitch_dur = np.random.randint(12, 30)
                if twitch_start + twitch_dur < 300:
                    twitch_env = np.sin(np.linspace(0, np.pi, twitch_dur))
                    baseline[twitch_start:twitch_start+twitch_dur] += (
                        twitch_env * np.random.uniform(0.3, 0.6)
                    )
            
            dummy[ch] = baseline
        
        # Extensor channels (5-9) - MODERATE ACTIVITY (maintain posture)
        for ch in range(5, 10):
            activation = np.ones(300) * np.random.uniform(0.9, 1.3)
            activation += 0.2 * np.sin(2 * np.pi * 2.2 * t)
            
            carrier = np.zeros(300)
            for _ in range(3):
                freq = np.random.uniform(16, 26)
                phase = np.random.uniform(0, 2 * np.pi)
                carrier += np.sin(2 * np.pi * freq * t + phase) / 3
            
            signal = activation * carrier * 0.48
            signal += np.random.randn(300) * 0.24
            
            dummy[ch] = signal
    
    # ===== ADD REALISTIC NOISE TO ALL CHANNELS =====
    for ch in range(10):
        signal_power = np.var(dummy[ch])
        
        # White Gaussian noise
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            white_noise = np.random.randn(300) * np.sqrt(noise_power)
        else:
            white_noise = np.random.randn(300) * 0.05
        
        # 50 Hz powerline
        powerline = np.random.uniform(0.015, 0.025) * np.sin(
            2 * np.pi * 50 * t + np.random.uniform(0, 2*np.pi)
        )
        
        # Motion artifacts
        motion = np.random.uniform(0.08, 0.14) * np.sin(
            2 * np.pi * np.random.uniform(0.4, 1.5) * t
        ) * np.random.rand()
        
        # HF noise
        hf_noise = np.random.randn(300) * 0.025
        
        # Baseline drift
        drift = np.random.uniform(-0.05, 0.05) * np.linspace(0, 1, 300)
        
        dummy[ch] += white_noise + powerline + motion + hf_noise + drift
    
    return dummy

# ========== PREDICTION ==========
_prediction_counter = 0
_recent_predictions = []

def predict_finger_movement():
    """
    Predict clench from 10-channel EMG window
    
    Returns: 
        [clench, False, False, False, False]
    """
    global _prediction_counter, _recent_predictions
    _prediction_counter += 1
    
    # Generate realistic dummy data
    # 70% open, 30% clench
    is_clench_phase = (_prediction_counter % 10) < 3
    dummy = generate_realistic_emg(clench=is_clench_phase, snr_db=18)
    
    # Extract features (90-D)
    feat = extract_features(dummy)
    
    # Scale and predict
    feat = scale_features(feat)
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 90)
    
    with torch.no_grad():
        prob = model(feat_tensor).squeeze().item()
        clench = prob > 0.5
        confidence = abs(prob - 0.5) * 2
        
        # Smoothing
        _recent_predictions.append(prob)
        if len(_recent_predictions) > 5:
            _recent_predictions.pop(0)
        
        avg_prob = np.mean(_recent_predictions)
        stable_clench = avg_prob > 0.5
        
        # Visual indicator
        status_emoji = "✊" if stable_clench else "✋"
        
        print(f"{status_emoji} [10-CH] Pred #{_prediction_counter:3d}: "
              f"prob={prob:.3f}, avg={avg_prob:.3f}, "
              f"clench={stable_clench}, conf={confidence:.2f}, "
              f"actual={'CLENCH' if is_clench_phase else 'OPEN  '}")
    
    return [stable_clench, False, False, False, False]


# ========== TESTING ==========
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing 10-Channel EMG Clench Detection (Dummy Data)")
    print("="*70)
    print(f"Model loaded: {scaler is not None}")
    print("="*70 + "\n")
    
    print("Running 30 predictions (you should see both ✊ and ✋):\n")
    
    for i in range(30):
        result = predict_finger_movement()
        
        import time
        time.sleep(0.1)
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print(f"Total predictions: {_prediction_counter}")
    print("="*70)