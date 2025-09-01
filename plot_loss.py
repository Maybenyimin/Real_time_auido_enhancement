import csv
import matplotlib.pyplot as plt

def read_log(path):
    epochs, train, val = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return epochs, train, val

def plot(csv_file, out_file, title):
    e, tr, va = read_log(csv_file)

    plt.figure(figsize=(9,6))
    plt.plot(e, tr, "--", label="Train Loss")
    plt.plot(e, va, "-",  label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (−SI-SNR)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot("logs/baseline_train_log.csv",
         "logs/loss_curves_baseline.png",
         "Baseline Conv-TasNet: Training & Validation Loss")
