"""Inference-only replacement for upstream's training/Lightning base class."""
import torch


class BaseLightningClass(torch.nn.Module):
    def save_hyperparameters(self, **kwargs):
        pass

    def update_data_statistics(self, data_statistics):
        stats = data_statistics or {'mel_mean': 0.0, 'mel_std': 1.0}
        self.register_buffer('mel_mean', torch.tensor(stats['mel_mean']))
        self.register_buffer('mel_std', torch.tensor(stats['mel_std']))
