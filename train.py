from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "outputs" / "train_log.txt"


def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def train_branch(model, Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, bs=64, seed=42,
                 device="cpu", log_path=DEFAULT_LOG_PATH):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    model.to(device)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)),
                     dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=bs, shuffle=True)
    best, best_state = -1, None
    logs = []
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(Xva).to(device)).argmax(1).cpu().numpy()
        f1 = macro_f1(yva, pv)
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
        line = f"epoch {ep + 1:02d}  val macro-F1 {f1:.4f}"
        print(line)
        logs.append(line)
    logs.append(f"best val macro-F1 {best:.4f}")
    if log_path is not None:
        out_file = Path(log_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(logs) + "\n", encoding="utf-8")
    model.load_state_dict(best_state)
    return model, best
