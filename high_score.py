import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data         import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import f1_score
from torch.nn.utils.parametrizations import weight_norm

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512,256,128), n_classes=10, p_dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(p_dropout)]
            prev = h
        layers += [weight_norm(nn.Linear(prev, n_classes, bias=False))]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def main():
    # 1) load raw
    df = pd.read_csv("fulltrain.txt", header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    X  = df.iloc[:, :-1].fillna(df.mean())
    y  = pd.read_csv("fulllabels.txt", header=None).to_numpy().ravel()

    # 2) split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) scale (fit on train only)
    scaler = StandardScaler().fit(X_tr)
    joblib.dump(scaler, "scaler.joblib")
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)

    best_f1, best_k = 0, None
    best_state, best_sel = None, None

    # 4) loop over k in feature‐selection
    for k in range(5, X_tr.shape[1]+1, 2):
        # select top-k
        selector = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
        Xtr_sel  = selector.transform(X_tr)
        Xval_sel = selector.transform(X_val)

        # optional: class balance via WeightedRandomSampler
        counts = np.bincount(y_tr)
        weights = 1.0/counts
        samp_wts = weights[y_tr]
        sampler = WeightedRandomSampler(samp_wts, len(samp_wts), replacement=True)

        # DataLoaders
        train_ds = TensorDataset(torch.from_numpy(Xtr_sel).float(),
                                 torch.from_numpy(y_tr).long())
        train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
        val_loader   = DataLoader(TensorDataset(
                                  torch.from_numpy(Xval_sel).float(),
                                  torch.from_numpy(y_val).long()),
                                  batch_size=256, shuffle=False)

        # build model + optimizer + scheduler + loss
        model     = MLP(input_dim=k, hidden_dims=(512,256,128), n_classes=len(np.unique(y)))
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()

        # early‐stop loop
        local_best, wait = 0, 0
        for epoch in range(1, 201):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward(); optimizer.step()

            # validate
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
                    trues.append(yb.numpy())
            preds = np.concatenate(preds); trues = np.concatenate(trues)
            f1    = f1_score(trues, preds, average="weighted")
            scheduler.step(f1)

            if f1 > local_best:
                local_best = f1; wait = 0
            else:
                wait += 1
                if wait >= 20:
                    break

        print(f"k={k} → val F1={local_best:.4f}")
        if local_best > best_f1:
            best_f1, best_k = local_best, k
            best_state      = model.state_dict()
            best_sel        = selector

    # 5) dump best artifacts
    print(f"Best overall: k={best_k}, F1={best_f1:.4f}")
    joblib.dump(best_sel,        "selector.joblib")
    joblib.dump({
        'input_dim': best_k,
        'hidden_dims': (512, 256, 128),
        'n_classes': len(np.unique(y))
    }, "model_config.joblib")

    torch.save(best_state, "model_state.pt")  # use best_state, not model.state_dict()

if __name__ == "__main__":
    main()