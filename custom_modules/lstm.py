import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

pen = 1.3 # 1.0 < pen < 2.0 (1.0 for none)
pen2 = 0.02 # 0.0 < pen2 < pen - 1 (0.00001 for none)
penMatrix = torch.tensor([
    [2.0 - pen, 1.0 - pen2, pen + pen2],
    [1.0 + pen2/2, 1.0 - pen2, 1.0 + pen2/2],
    [pen + pen2, 1.0 - pen2, 2.0 - pen],
], dtype=torch.float32, device="cpu")
penMatrixNp = penMatrix.numpy()

def costScore(y_true, y_preds):
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1, 2]).astype(float) # 3x3 numpy array (real, predicted)
    cm /= cm.sum() # still 3x3 but now sums to 1
    cost = (cm * penMatrixNp).sum() # elementwise multiply by costs, then sum to scalar value

    classFreqs = cm.sum(axis=1) # true distribution
    bestCase = (classFreqs * penMatrixNp.diagonal()).sum() # all minimum multipliers
    worstCase = (classFreqs * np.fliplr(penMatrixNp).diagonal()).sum() # all maximum multipliers
    score = 1 - ((cost - bestCase) / (worstCase - bestCase)) # normalise to 0-1 (1 is best)
    # penalise collapse
    predFreqs = cm.sum(axis=0) # prediction distribution
    collapsePenalty = max(0, predFreqs.max() - 0.5) # 0 if no class above 0.5
    return score - collapsePenalty

def optimiserBundle(model, labels, device, optimiser_name, learning_rate, weight_decay, scheduler_patience=5):
    classCounts = np.bincount(labels.astype(int)) # no. of each class
    classWeights = 1.0 / classCounts # majority class => smaller weight and vice versa
    classWeights = (classWeights / classWeights.sum()) * len(classWeights) # normalise
    classWeights = classWeights * np.array([1.2, 1.0, 1.2]) # for manual adjusting (all 1.0 for default)
    weightsTensor = torch.tensor(classWeights, dtype=torch.float32, device=device) # penalise mistakes on minority classes more
    
    class CostSensitiveLoss(torch.nn.Module):
        def __init__(self, cost_matrix, class_weights):
            super().__init__()
            self.cost_matrix = cost_matrix.to(device)
            self.class_weights = class_weights.to(device)

        def forward(self, logits, targets):
            # logits: raw scores (batch_size, 3)
            # targets: true classes (batch_size,)
            probs = torch.softmax(logits, dim=1)
            costs = self.cost_matrix[targets] # get respective cost row for each sample
            cost_weighted_probs = (costs * probs).sum(dim=1) # multiply each probability by respective coeff, sum to scalar per sample
            ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none") # normal crossentropyloss, per sample loss
            loss = (ce * cost_weighted_probs).mean() # apply multiplier to standard loss (random: 1.0)
            return loss

    criterion = CostSensitiveLoss(penMatrix, weightsTensor) # function to minimise
    criterion = torch.nn.CrossEntropyLoss(weight=weightsTensor) # standard CE: comment out when not needed
    optimiserClass = {"AdamW": torch.optim.AdamW, "RMSprop": torch.optim.RMSprop}[optimiser_name]
    optimiser = optimiserClass(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=scheduler_patience, min_lr=1e-6)

    return criterion, optimiser, scheduler, classWeights

def classBuilder():
    class ForexRNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
            super(ForexRNN, self).__init__()
            # LSTM layer: takes 3D tensor as input (batch_size, timesteps, features)
            self.lstm = torch.nn.LSTM(
                input_size=input_size, # no. of features per datapoint
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True, # set batch size as first dimension of input tensor
                dropout=dropout if num_layers > 1 else 0
            )
            # Output layer (maps final pattern produced by LSTM to actual prediction) (fully connected)
            self.fc = torch.nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # x is 3D tensor (batch_size, timesteps, features)
            lstmOutput, (hidden, cell) = self.lstm(x)
            # lstmOutput: hidden state of EVERY timestep for the LAST layer only (batch_size, timesteps, hidden_size)
            # hidden: final hidden state (LAST timestep) for EVERY layer (layers, batch_size, hidden_size)
            # cell: similar to hidden but contains cell state instead of hidden state
            lastTimestep = lstmOutput[:, -1, :] # slice out last timestep across all samples and neurons
            return self.fc(lastTimestep) # map to prediction (batch_size, output size)

    class ForexHybrid(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, lstm_dropout, output_size,
                    num_filters, kernel_size):
            super(ForexHybrid, self).__init__()
            # CNN layers: takes 3D tensor as input (batch_size, channels, length)
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                        kernel_size=kernel_size, padding=kernel_size//2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_filters), # normalise before passing to LSTM
                torch.nn.Dropout(dropout)
            )
            # LSTM layers
            self.lstm = torch.nn.LSTM(
                input_size=num_filters, # takes CNN output
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout if num_layers > 1 else 0
            )
            # Output layer (maps final pattern produced by LSTM to actual prediction) (fully connected)
            self.fc = torch.nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # x is 3D tensor (batch_size, timesteps, features)
            # CNN
            x = x.permute(0, 2, 1) # (batch, features, timesteps)
            x = self.cnn(x)
            x = x.permute(0, 2, 1) # (batch, timesteps, num_filters)
            # LSTM
            lstmOutput, (hidden, cell) = self.lstm(x)
            # lstmOutput: hidden state of EVERY timestep for the LAST layer only (batch_size, timesteps, hidden_size)
            # hidden: final hidden state (LAST timestep) for EVERY layer (layers, batch_size, hidden_size)
            # cell: similar to hidden but contains cell state instead of hidden state
            # FC
            lastTimestep = lstmOutput[:, -1, :] # slice out last timestep across all samples and neurons
            return self.fc(lastTimestep) # map to prediction (batch_size, output size)
    
    return ForexRNN, ForexHybrid

def numParams(model):
    total = sum(p.numel() for p in model.parameters()) # total no. of elements
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable