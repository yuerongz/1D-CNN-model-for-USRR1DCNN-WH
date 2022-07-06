import numpy as np
import torch
import torch.nn as nn

class ConvoModel2RLs(nn.Module):
    def __init__(self, model_structure, seq_h):
        super(ConvoModel2RLs, self).__init__()
        convo_1_kernel = 4
        pool_1_kernel = 3
        self.convo_1 = nn.Conv1d(model_structure[0], model_structure[1], convo_1_kernel)
        self.pooling_1 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.convo_2 = nn.Conv1d(model_structure[1], model_structure[1], convo_1_kernel)
        self.pooling_2 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.dim_past_convo = lambda dim_in: int(np.ceil((dim_in - (convo_1_kernel-1))/pool_1_kernel))
        flat_dim = self.dim_past_convo(self.dim_past_convo(seq_h * 4)) * model_structure[1]
        self.hidden_1 = nn.Linear(flat_dim, model_structure[-3])
        self.lyr_out = nn.Linear(model_structure[-3], model_structure[-1])
        self.lrelu = nn.LeakyReLU()

    def forward(self, src, dw_filter, ref, trainning_with_dw_mask=True):
        src = self.convo_1(src.transpose(1, 2))
        src = torch.tanh(self.pooling_1(src))
        src = self.convo_2(src)
        src = self.lrelu(self.pooling_2(src))
        src = self.lrelu(self.hidden_1(src.view(src.size()[0], 1, -1)))
        src = self.lyr_out(src).squeeze(1)
        if trainning_with_dw_mask:
            src = ref * dw_filter + (-dw_filter + 1) * src + torch.relu(src*dw_filter - ref * dw_filter)
        return src
