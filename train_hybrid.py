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
from models.hybrid_model import HybridModel
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
    def __init__(self, path: str):
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

def normalize_waveform(wave: torch.Tensor) -> torch.Tensor:
    return wave / (wave.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

def match_lengths(a: torch.Tensor, b: torch.Tensor):
    min_len = min(a.shape[-1], b.shape[-1])
    return a[..., :min_len], b[..., :min_len]

def create_dataloaders(config):
    train_set = SpeechDataset(
        config["dataset"]["train_noisy"],
        config["dataset"]["train_clean"],
        segment_length=config["segment_length"]
    )
    val_set = SpeechDataset(
        config["dataset"]["test_noisy"],
        config["dataset"]["test_clean"],
        segment_length=config["segment_length"]
    )
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    return train_loader, val_loader

def get_model(config):
    model = HybridModel(config)
    return model.to(DEVICE)

def get_optimizer(config, model):
    return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# ---------------- TRAINING ----------------
def train_one_epoch(model, loader, loss_fn, optimizer, epoch, sample_rate):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for i, (noisy, clean) in enumerate(progress):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        noisy = normalize_waveform(noisy)
        clean = normalize_waveform(clean)

        optimizer.zero_grad(set_to_none=True)
        enhanced = model(noisy)
        enhanced, clean = match_lengths(enhanced, clean)

        loss = loss_fn(enhanced, clean)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        avg_loss = total_loss / (i + 1)
        progress.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Avg Epoch Loss": f"{avg_loss:.4f}"})

        # Save one audio example every 10 epochs
        if epoch % 10 == 0 and i == 0:
            os.makedirs("output_samples", exist_ok=True)
            sf.write(f"output_samples/hybrid_epoch{epoch}.wav",
                     enhanced[0].detach().cpu().numpy(), sample_rate)

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        noisy = normalize_waveform(noisy)
        clean = normalize_waveform(clean)

        enhanced = model(noisy)
        enhanced, clean = match_lengths(enhanced, clean)

        loss = loss_fn(enhanced, clean)
        total_loss += float(loss.item())
    return total_loss / len(loader)

def train(config_path: str):
    config = load_config(config_path)
    train_loader, val_loader = create_dataloaders(config)
    model = get_model(config)

    # (Optional) quick shape sanity check
    dummy = torch.randn(2, int(config["sample_rate"] * config["segment_length"])).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)
    print(f"[DEBUG] Hybrid forward: {tuple(dummy.shape)} → {tuple(out.shape)}")

    optimizer = get_optimizer(config, model)
    loss_fn = si_snr_loss
    sample_rate = config["sample_rate"]

    os.makedirs("checkpoints", exist_ok=True)
    logger = CSVLogger("logs/hybrid_train_log.csv")

    best_val = float("inf")
    patience = int(config["early_stopping_patience"])
    wait = 0

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, sample_rate)
        va = validate(model, val_loader, loss_fn)

        secs = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"✅ Epoch {epoch:02d} | Train {tr:.4f} | Val {va:.4f} | LR {lr:.2e} | {secs:.2f}s")

        # log row
        logger.write(EpochLog(epoch=epoch, train_loss=float(tr), val_loss=float(va), lr=float(lr), secs=float(secs)))

        # early stopping + checkpoint
        if va < best_val:
            best_val = va
            wait = 0
            ckpt_path = f"checkpoints/hybrid_best_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✔️ Saved new best: {ckpt_path}")
        else:
            wait += 1
            if wait >= patience:
                print("🛑 Early stopping triggered.")
                break

if __name__ == "__main__":
    # Use your hybrid config path here:
    train("/content/real-time-speech-enhancement/config.yaml")
