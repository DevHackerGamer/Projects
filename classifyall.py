import pandas as pd
import joblib
import torch
import numpy as np
from high_score import MLP

def load_and_clean(path):
    """
    Read raw testdata.txt (no header), drop the last placeholder column,
    coerce to numeric & fill NaNs with column means.
    """
    df = pd.read_csv(path, header=None)
    X  = df.iloc[:, :-1]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(X[c].mean())
    return X.to_numpy()

def main():
    # 1) load artifacts
    scaler     = joblib.load("scaler.joblib")
    selector   = joblib.load("selector.joblib")
    state_dict = torch.load("model_state.pt", map_location=torch.device("cpu"))
    
 # infer number of output classes from the last layer’s weight_norm g-vector
    n_classes = None
    for key, val in state_dict.items():
        if key.endswith("net.9.parametrizations.weight.original0"):
            # this vector has shape [n_classes]
            n_classes = val.shape[0]
            break
    assert n_classes is not None, "could not find final g-vector in checkpoint"


    # 2) load & preprocess unseen data
    X = load_and_clean("testdata.txt")
    X = scaler.transform(X)
    X = selector.transform(X)
    X_t = torch.from_numpy(X).float()

    # 3) build model and load weights
    config = joblib.load("model_config.joblib")
    model = MLP(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        n_classes=config['n_classes']
    )    
    model.load_state_dict(state_dict)
    model.eval()

    # 4) predict
    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=1).cpu().numpy()

    # 5) save predictions
    pd.DataFrame(preds).to_csv(
        "predlabels.txt",
        index=False,
        header=False
    )

if __name__ == "__main__":
    main()