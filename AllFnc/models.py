import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import filtfilt
import selfeeg
from selfeeg.augmentation import get_filter_coeff
from selfeeg.models import (
    ConstrainedConv2d,
    ConstrainedConv1d,
    ConstrainedDense,
    DepthwiseConv2d,
    SeparableConv2d,
)


__all__ = [
    "HybridNetEncoder",
    "HybridNet",
]

# -----------------------------
#          HybridNet
# -----------------------------
class HybridNetEncoder(nn.Module):

    def __init__(
        self,
        Chans,
        F1 = 64,
        F2 = 128,
        lstm  = 32,
        kernLength1 = 11,
        kernLength2 = 3,
        pool  = 4,
        stridePool = 2,
        dropRate = 0.1,
        ELUAlpha = 0,
        batchMomentum = 0.9,
        max_norm = None,
    ):
        
        super(HybridNetEncoder, self).__init__()

        self.blck1 = nn.Sequential(
            ConstrainedConv1d(Chans, F1, kernLength1, padding = 'same', max_norm = max_norm),
            nn.BatchNorm1d(F1, momentum = batchMomentum),
            nn.ELU(ELUAlpha)
        )
        
        self.pool1 = nn.MaxPool1d(pool, stridePool)
        self.drop1 = nn.Dropout1d(dropRate)
        
        self.blck2 = nn.Sequential(
            ConstrainedConv1d(F1, F2, kernLength2, padding = 'same', max_norm = max_norm),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha)
        )

        self.blck3 = nn.Sequential(
            ConstrainedConv1d(F2, F2, kernLength2, padding = 'same', max_norm = max_norm),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha)
        )

        self.blck4 = nn.Sequential(
            ConstrainedConv1d(F2, F2, kernLength2, padding = 'same', max_norm = max_norm),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha)
        )

        self.pool2 = nn.MaxPool1d(pool, stridePool)
        self.drop2 = nn.Dropout1d(dropRate)

        self.lstm1 = nn.LSTM(input_size = F2, hidden_size = lstm, num_layers = 1)

    def forward(self, x):

        x = self.blck1(x)
        
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.blck2(x)
        x = self.blck3(x)
        x = self.blck4(x)
        
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.permute(x, (2, 0, 1))
        out, (ht, ct) = self.lstm1(x)
        
        return ht[-1]


class HybridNet(nn.Module):

    def __init__(
        self,
        nb_classes,
        Chans,
        F1 = 64,
        F2 = 128,
        lstm  = 32,
        dense = 64,
        kernLength1 = 11,
        kernLength2 = 3,
        pool  = 4,
        stridePool = 2,
        dropRate = 0.1,
        ELUAlpha = 0,
        batchMomentum = 0.9,
        max_norm = 1.0,
        classifier: nn.Module = None,
        return_logits: bool = True
    ):

        super(HybridNet, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        
        # Encoder
        self.encoder = HybridNetEncoder(
            Chans,
            F1,
            F2,
            lstm,
            kernLength1,
            kernLength2,
            pool,
            stridePool,
            dropRate,
            ELUAlpha,
            batchMomentum,
            max_norm
        )

        # Head
        if classifier is None:
            self.head = nn.Sequential(
                nn.Dropout1d(dropRate),
                nn.Linear(lstm, dense),
                nn.Linear(dense, 1 if nb_classes <= 2 else nb_classes)
            )
        else:
            self.head = classifier


    def forward(self, x):

        x = self.encoder(x)
        x = self.head(x)
        if not self.return_logits:
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x
