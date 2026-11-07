# project/Telugu/model.py
import torch
import torch.nn as nn
import yaml

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2
    def forward(self, x): 
        return x.transpose(self.dim1, self.dim2)

class VariancePredictor(nn.Module):
    """
    Variance Predictor Block for predicting duration, pitch, and energy profiles.
    Used by the Telugu variance adapter variance layers.
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = config["model"]["encoder_hidden"]
        filter_size = config["model"]["variance_predictor_filter_size"]
        kernel_size = config["model"]["variance_predictor_kernel_size"]
        dropout = config["model"]["dropout"]

        self.layers = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(hidden_dim, filter_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(filter_size),
            nn.Dropout(dropout),
            nn.Conv1d(filter_size, filter_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(filter_size),
            nn.Dropout(dropout),
            Transpose(1, 2),
            nn.Linear(filter_size, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

class FastSpeech2Telugu(nn.Module):
    """
    FastSpeech 2 Core Architecture customized for Telugu Unicode Symbol Vocabularies.
    Includes Character Embedding, Variance Adaptor, and Output Projection.
    """
    def __init__(self, config, vocab_size=150):
        super().__init__()
        hidden_dim = config["model"]["encoder_hidden"]
        
        # Telugu Unicode Embedding Layer (Handles Telugu Syllabary Sequences)
        self.encoder_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        # Variance Adaptor Blocks
        self.duration_predictor = VariancePredictor(config)
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)

        # Quantized feature embeddings mapping continuous curves to structural states
        self.pitch_embedding = nn.Embedding(256, hidden_dim)
        self.energy_embedding = nn.Embedding(256, hidden_dim)
        
        # Mel Spectrogram Projection Module (Maps hidden representations into acoustic dimensions)
        self.mel_linear = nn.Linear(hidden_dim, config["audio"]["n_mel_channels"])

    def length_regulator(self, x, durations):
        """Expands character level hidden states along time frames using phonetic durations."""
        output = []
        for i, d in enumerate(durations):
            expanded = x[i].repeat_interleave(torch.clamp(d.long(), min=0), dim=0)
            output.append(expanded)
        return nn.utils.rnn.pad_sequence(output, batch_first=True)

    def forward(self, src_seq, duration_target=None, pitch_target=None, energy_target=None):
        # 1. Transform Telugu token sequences into character feature representations
        x = self.encoder_embedding(src_seq)
        
        # 2. Extract and estimate variances
        log_duration_pred = self.duration_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)

        # 3. Apply standard duration adjustments (Branch path mapping Training vs Inference)
        if duration_target is None:
            durations = torch.clamp(torch.exp(log_duration_pred) - 1, min=0).round()
            x = self.length_regulator(x, durations)
        else:
            x = self.length_regulator(x, duration_target)

        # 4. Inject physical targets properties safely
        if pitch_target is not None and energy_target is not None:
            # Map scalar arrays into discrete bins dynamically
            p_bins = torch.bucketize(pitch_target, torch.linspace(0, 500, 256).to(x.device))
            e_bins = torch.bucketize(energy_target, torch.linspace(0, 1, 256).to(x.device))
            
            p_embed = self.pitch_embedding(torch.clamp(p_bins, 0, 255))
            e_embed = self.energy_embedding(torch.clamp(e_bins, 0, 255))
            
            # Align shapes if length regulation expanded structural dimensions
            if x.shape[1] == p_embed.shape[1]:
                x = x + p_embed + e_embed

        # 5. Project final features to 80-channel Mel target dimension vectors
        mel_output = self.mel_linear(x)
        
        return mel_output, log_duration_pred, pitch_pred, energy_pred

# Alias so inference.py can import VarianceAdaptor consistently across both languages
VarianceAdaptor = FastSpeech2Telugu

if __name__ == "__main__":
    # Internal component sanity testing validation check
    mock_config = {
        "model": {
            "encoder_hidden": 256,
            "variance_predictor_filter_size": 256,
            "variance_predictor_kernel_size": 3,
            "dropout": 0.1
        },
        "audio": {
            "n_mel_channels": 80
        }
    }
    
    print("Initializing Telugu Model testing framework...")
    model = FastSpeech2Telugu(mock_config)
    mock_input = torch.randint(1, 100, (2, 15))  # Batch size 2, Sequence length 15 tokens
    mel_out, dur_out, p_out, e_out = model(mock_input)
    
    print(f"Telugu Model successfully initialized. Mel Tensor shape: {mel_out.shape}")
