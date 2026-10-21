import torch
import torch.nn as nn
from transformers import WavLMModel

# WavLM hat 12 Schichten (Transormer Blöcke) + Eingabe Schicht = 13 und Aus einem 4-Sekunden-Clip erzeugt jede Schicht pro 20-ms-Frame einen Vektor aus 768 Zahlen, insgesamt 199 Vektoren pro Schicht.
class SER(nn.Module):
    def __init__(self, n_classes=7, name="microsoft/wavlm-base-plus", dropout=0.3):
        super().__init__()
        self.encoder = WavLMModel.from_pretrained(name)
        hidden = self.encoder.config.hidden_size          # 768
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.encoder(x).last_hidden_state             # (B, T, 768)
        pooled = h.mean(dim=1)                            # (B, 768)
        return self.head(pooled)                          # (B, n_classes)



if __name__ == "__main__":
    model = SER()
    x = torch.randn(2, 64000)
    print(model(x).shape)                                    # torch.Size([2, 7])

    total = sum(p.numel() for p in model.parameters())
    train_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.freeze_encoder()
    train_head = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total, train_all, train_head)