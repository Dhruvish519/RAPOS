"""
NumPy inference for the RAPOS stock LSTM.

The model is trained with PyTorch (train_lstm.py), but serving it with PyTorch
costs ~300 MB of RAM - more than Render's free plan allows alongside OpenCV.
The forward pass of a 2-layer LSTM is a few matrix products, so the trained
weights are exported once (export_lstm_weights.py -> lstm_weights.npz) and run
here with NumPy. Outputs match PyTorch to ~1e-6 (see tests).

PyTorch LSTM equations per layer (gate order i, f, g, o):
    gates = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh
    i, f, o = sigmoid(...),  g = tanh(...)
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)
Output = Linear(h_T of the last layer). Dropout is inactive at inference.
"""
from __future__ import annotations

import os

import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "lstm_weights.npz")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    def __init__(self, path: str = WEIGHTS_PATH):
        w = np.load(path)
        self.num_layers = int(w["num_layers"])
        self.hidden_size = int(w["hidden_size"])
        self.sequence_length = int(w["sequence_length"])
        self.layers = [(w[f"weight_ih_l{k}"], w[f"weight_hh_l{k}"],
                        w[f"bias_ih_l{k}"] + w[f"bias_hh_l{k}"]) for k in range(self.num_layers)]
        self.fc_w, self.fc_b = w["fc_weight"], w["fc_bias"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, time, features) -> (batch, outputs)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x[None]
        seq = x
        H = self.hidden_size
        for w_ih, w_hh, b in self.layers:
            n, T, _ = seq.shape
            h = np.zeros((n, H))
            c = np.zeros((n, H))
            pre = seq @ w_ih.T + b              # input part for all steps at once
            outs = np.empty((n, T, H))
            for t in range(T):
                g = pre[:, t] + h @ w_hh.T
                i, f = _sigmoid(g[:, :H]), _sigmoid(g[:, H:2 * H])
                gg, o = np.tanh(g[:, 2 * H:3 * H]), _sigmoid(g[:, 3 * H:])
                c = f * c + i * gg
                h = o * np.tanh(c)
                outs[:, t] = h
            seq = outs
        return seq[:, -1] @ self.fc_w.T + self.fc_b


_model: NumpyLSTM | None = None


def load(path: str = WEIGHTS_PATH) -> NumpyLSTM | None:
    global _model
    if _model is None and os.path.exists(path):
        _model = NumpyLSTM(path)
    return _model
