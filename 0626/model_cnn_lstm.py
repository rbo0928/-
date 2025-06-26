import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_output=256, lstm_hidden=128):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=8192, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        cnn_feat = self.cnn(x)
        cnn_feat = cnn_feat.view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_feat)
        output = self.fc(lstm_out[:, -1, :])
        return output.squeeze(1)