"""The cached-sequence CNN input contract, shared with deployment."""
import numpy as np
import torch


def windows_from_sequence(sequence, win_len=100, hop_len=50, min_win=30):
    """Trim cache padding, edge-pad short turns, then z-score each window."""
    sequence = np.asarray(sequence)
    non_padding = np.any(np.abs(sequence) > 1e-12, axis=1)
    length = int(np.flatnonzero(non_padding)[-1] + 1) if np.any(non_padding) else 0
    if length < min_win:
        return []
    sequence = np.asarray(sequence[:length], dtype=np.float32)
    if length < win_len:
        sequence = np.pad(sequence, ((0, win_len - length), (0, 0)), mode="edge")
    windows = []
    for start in range(0, len(sequence) - win_len + 1, hop_len):
        window = sequence[start:start + win_len].T.copy()
        window = (window - window.mean(axis=1, keepdims=True)) / (
            window.std(axis=1, keepdims=True) + 1e-8
        )
        if np.isfinite(window).all():
            windows.append(torch.from_numpy(window))
    return windows
