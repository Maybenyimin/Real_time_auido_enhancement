import os
import yaml
import torch
import time
import soundfile as sf
import csv
from dataclasses import dataclass, asdict
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import SpeechDataset
from models.conv_tasnet import ConvTasNet
from utils.losses import si_snr_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- CSV LOGGER ----------------
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    secs: float

class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=EpochLog.__annotations__.keys())
                w.writeheader()

    def write(self, rec: EpochLog):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EpochLog.__annotations__.keys())
            w.writerow(asdict(rec))


# ---------------- HELPERS ----------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def normalize_waveform(wave):
    return wave / (wave.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

def match_lengths(a, b):
    min_len = min(a.shape[-1], b.shape[-1])
    return a[..., :min_len], b[..., :min_len]

def create_dataloaders(config):
    train_set = SpeechDataset(
        config["dataset"]["train_noisy"], config["dataset"]["train_clean"],
        segment_length=config["segment_length"]
    )
    test_set = SpeechDataset(
        config["dataset"]["test_noisy"], config["dataset"]["test_clean"],
        segment_length=config["segment_length"]
    )
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, test_loader

def get_model(config):
    model = ConvTasNet(
        enc_kernel_size=config["encoder"]["kernel_size"],
        enc_stride=config["encoder"]["stride"],
        enc_out_channels=config["encoder"]["out_channels"],
        num_tcn_blocks=config["tcn"]["num_blocks"],
        tcn_kernel_size=config["tcn"]["kernel_size"],
        use_post_conv=True
    )
    return model.to(DEVICE)

def get_optimizer(config, model):
    return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])


# ---------------- TRAINING ----------------
def train_one_epoch(model, loader, loss_fn, optimizer, epoch, sample_rate):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for i, (noisy, clean) in enumerate(progress):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        # Normalize
        noisy = normalize_waveform(noisy)
        clean = normalize_waveform(clean)

        optimizer.zero_grad()
        enhanced = model(noisy)
        enhanced, clean = match_lengths(enhanced, clean)

        loss = loss_fn(enhanced, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        progress.set_postfix({"Batch Loss": loss.item(), "Avg Epoch Loss": avg_loss})

        # Save sample every 10 epochs
        if epoch % 10 == 0 and i == 0:
            os.makedirs("output_samples", exist_ok=True)
            out_path = f"output_samples/enhanced_epoch{epoch}.wav"
            sf.write(out_path, enhanced[0].detach().cpu().numpy(), sample_rate)

    return total_loss / len(loader)

def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            noisy = normalize_waveform(noisy)
            clean = normalize_waveform(clean)

            enhanced = model(noisy)
            enhanced, clean = match_lengths(enhanced, clean)
            loss = loss_fn(enhanced, clean)
            total_loss += loss.item()
    return total_loss / len(loader)


def train(config_path):
    config = load_config(config_path)
    train_loader, val_loader = create_dataloaders(config)
    model = get_model(config)
    optimizer = get_optimizer(config, model)
    loss_fn = si_snr_loss
    sample_rate = config["sample_rate"]

    os.makedirs("checkpoints", exist_ok=True)
    logger = CSVLogger("logs/baseline_train_log.csv")

    best_val_loss = float("inf")
    patience = config["early_stopping_patience"]
    counter = 0

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch + 1, sample_rate)
        val_loss = validate(model, val_loader, loss_fn)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(f"✅ Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | "
              f"LR = {lr:.2e} | Time: {elapsed:.2f}s")

        logger.write(EpochLog(epoch=epoch+1,
                              train_loss=float(train_loss),
                              val_loss=float(val_loss),
                              lr=float(lr),
                              secs=float(elapsed)))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print("✔️ Saved new best model.")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping triggered.")
                break


if __name__ == "__main__":
    train("/content/real-time-speech-enhancement/config_baseline1.yaml")