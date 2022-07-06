import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, model_structure):
        super(LSTMModel, self).__init__()
        self.biis = nn.Linear(model_structure[0], model_structure[1])
        self.lstm = nn.LSTM(model_structure[1], model_structure[2], batch_first=True, bidirectional=False)
        self.hidden_lyr = nn.Linear(model_structure[2], model_structure[-2])
        self.lyr_out = nn.Linear(*model_structure[-2:])
        self.lrelu = nn.LeakyReLU()

    def forward(self, src, dw_filter, ref, trainning_with_dw_mask=True):
        src = self.lrelu(self.biis(src))    #(N, L, D)
        _, (src, _) = self.lstm(src)
        src = src.transpose(0, 1).reshape(-1, 1, src.size()[-1])    #(N, 1, 2 * H2out)
        src = self.lrelu(self.hidden_lyr(src))
        src = self.lyr_out(src).squeeze(1)
        if trainning_with_dw_mask:
            src = ref * dw_filter + (-dw_filter + 1) * src + torch.relu(src*dw_filter - ref * dw_filter)
        return src
