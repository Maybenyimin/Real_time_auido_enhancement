import csv
import matplotlib.pyplot as plt

def read_log(path):
    e,t,v = [],[],[]
    with open(path,"r") as f:
        r = csv.DictReader(f)
        for row in r:
            e.append(int(row["epoch"]))
            t.append(float(row["train_loss"]))
            v.append(float(row["val_loss"]))
    return e,t,v

if __name__ == "__main__":
    csv_path = "logs/hybrid_train_log.csv"
    out_png  = "logs/loss_curves_hybrid.png"
    epochs, tr, va = read_log(csv_path)

    plt.figure(figsize=(9,6))
    plt.plot(epochs, tr, "--", label="Train Loss")
    plt.plot(epochs, va, "-",  label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (−SI-SNR)")
    plt.title("Hybrid (Conv-TasNet + Causal SE-Conformer): Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")
