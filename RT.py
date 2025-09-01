import os
import yaml
import torch
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import time
from tqdm import tqdm

from models.conv_tasnet import ConvTasNet
from utils.evaluation import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(config):
    model = ConvTasNet(
        enc_kernel_size=config["encoder"]["kernel_size"],
        enc_stride=config["encoder"]["stride"],
        enc_out_channels=config["encoder"]["out_channels"],
        num_tcn_blocks=config["tcn"]["num_blocks"],
        tcn_kernel_size=config["tcn"]["kernel_size"],
        use_post_conv=True
    )
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def save_visualizations(noisy, clean, enhanced, sr, fname, output_dir):
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))

    # Waveforms
    plt.subplot(3, 2, 1)
    plt.title("Noisy - Waveform")
    librosa.display.waveshow(noisy, sr=sr)

    plt.subplot(3, 2, 3)
    plt.title("Clean - Waveform")
    librosa.display.waveshow(clean, sr=sr)

    plt.subplot(3, 2, 5)
    plt.title("Enhanced - Waveform")
    librosa.display.waveshow(enhanced, sr=sr)

    # Spectrograms
    noisy_spec = librosa.amplitude_to_db(np.abs(librosa.stft(noisy)), ref=np.max)
    clean_spec = librosa.amplitude_to_db(np.abs(librosa.stft(clean)), ref=np.max)
    enhanced_spec = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)

    plt.subplot(3, 2, 2)
    plt.title("Noisy - Spectrogram")
    librosa.display.specshow(noisy_spec, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

    plt.subplot(3, 2, 4)
    plt.title("Clean - Spectrogram")
    librosa.display.specshow(clean_spec, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

    plt.subplot(3, 2, 6)
    plt.title("Enhanced - Spectrogram")
    librosa.display.specshow(enhanced_spec, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{fname.replace('.wav', '')}_viz.png"))
    plt.close()

def evaluate(config_path):
    config = load_config(config_path)
    model = load_model(config)

    noisy_dir = config["dataset"]["test_noisy"]
    clean_dir = config["dataset"]["test_clean"]
    target_sr = config["sample_rate"]
    output_dir = "enhanced_outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)

    metrics_all = {"pesq": [], "stoi": [], "si_snr": [], "rtf": []}
    file_names = sorted([
        f for f in os.listdir(noisy_dir) if f.endswith(".wav") and f in os.listdir(clean_dir)
    ])

    for i, fname in enumerate(tqdm(file_names, desc="🔊 Enhancing")):
        noisy_path = os.path.join(noisy_dir, fname)
        clean_path = os.path.join(clean_dir, fname)

        noisy, sr = sf.read(noisy_path)
        clean, sr2 = sf.read(clean_path)

        if sr != target_sr:
            noisy = librosa.resample(noisy, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        if sr2 != target_sr:
            clean = librosa.resample(clean, orig_sr=sr2, target_sr=target_sr)
            sr2 = target_sr

        min_len = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_len].copy(), clean[:min_len].copy()

        noisy_tensor = torch.tensor(noisy / (np.max(np.abs(noisy)) + 1e-9), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        start_time = time.time()
        with torch.no_grad():
            enhanced = model(noisy_tensor).squeeze(0).cpu().numpy()
        inference_time = time.time() - start_time
        rtf = inference_time / (len(noisy) / sr)
        metrics_all["rtf"].append(rtf)

        enhanced /= np.max(np.abs(enhanced) + 1e-9)
        out_path = os.path.join(output_dir, f"enhanced_{i}_{fname}")
        sf.write(out_path, enhanced, sr)

        try:
            metrics = compute_metrics(enhanced.copy(), clean.copy(), sr)
            for k in ["pesq", "stoi", "si_snr"]:
                metrics_all[k].append(metrics[k])
        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")

        save_visualizations(noisy, clean, enhanced, sr, fname, output_dir)

    print("\n📊 Final Evaluation Results:")
    for k in ["pesq", "stoi", "si_snr", "rtf"]:
        if metrics_all[k]:
            avg = sum(metrics_all[k]) / len(metrics_all[k])
            print(f"✅ {k.upper()}: {avg:.4f}")
        else:
            print(f"⚠️ {k.upper()}: No valid results.")

if __name__ == "__main__":
    evaluate("config_baseline1.yaml")