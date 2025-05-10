import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # Input features: 12 piece types * 64 squares = 768 features
        self.feature_transforms = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, x):
        x = self.feature_transforms(x)
        x = self.output_layer(x)
        return x

def extract_features(board):
    """Convert board state to feature vector"""
    features = np.zeros(768, dtype=np.float32)
    
    # Map pieces to feature indices
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Extract features for each piece
    for piece, bb in board.bb.items():
        idx = piece_to_idx[piece]
        for sq in range(64):
            if bb & (1 << sq):
                features[idx * 64 + sq] = 1.0
                
    return torch.FloatTensor(features)

def evaluate_position(board, model):
    """Evaluate position using NNUE"""
    features = extract_features(board)
    with torch.no_grad():
        score = model(features)
    return score.item() * 1000  # Scale to centipawns 