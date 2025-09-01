import csv
import os
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
    os.makedirs("logs", exist_ok=True)
    base_csv = "logs/baseline_train_log.csv"
    hyb_csv  = "logs/hybrid_train_log.csv"
    out_png  = "logs/loss_curves_baseline_vs_hybrid.png"

    e_b, t_b, v_b = read_log(base_csv)
    e_h, t_h, v_h = read_log(hyb_csv)

    plt.figure(figsize=(10,6))
    plt.plot(e_b, t_b, "--", label="Baseline Train")
    plt.plot(e_b, v_b, "-",  label="Baseline Val")
    plt.plot(e_h, t_h, "--", label="Hybrid Train")
    plt.plot(e_h, v_h, "-",  label="Hybrid Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (−SI-SNR)")
    plt.title("Training & Validation Loss: Baseline vs Hybrid")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")
