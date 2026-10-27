"""
Export the trained PyTorch LSTM (lstm_model.pth) to lstm_weights.npz so the
API can run it with NumPy only. Run after train_lstm.py:

    python train_lstm.py && python export_lstm_weights.py
"""
import os

import numpy as np
import torch

HERE = os.path.dirname(__file__)


def export(src=os.path.join(HERE, "lstm_model.pth"), dst=os.path.join(HERE, "lstm_weights.npz")):
    ck = torch.load(src, map_location="cpu")
    sd = {k: v.numpy().astype(np.float64) for k, v in ck["model_state_dict"].items()}
    out = {"num_layers": ck["num_layers"], "hidden_size": ck["hidden_size"],
           "sequence_length": ck["sequence_length"],
           "fc_weight": sd["fc.weight"], "fc_bias": sd["fc.bias"]}
    for k in range(ck["num_layers"]):
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            out[f"{name}_l{k}"] = sd[f"lstm.{name}_l{k}"]
    np.savez_compressed(dst, **out)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    export()
