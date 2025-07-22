import torch
from torch.utils.data import Dataset, DataLoader
from data import WordVocab, Protein_dataset
from model import Protein_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.protein_processor import ProteinInference
import random

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128

# PPI model definition
class PPI(nn.Module):
    def __init__(self):
        super(PPI, self).__init__()
        # Protein graph + seq
        self.ligand_graph_model = Protein_feature_extraction(hidden_dim)
        self.receptor_graph_model = Protein_feature_extraction(hidden_dim)
        # Cross fusion module
        self.cross_attention = cross_attention(hidden_dim)
        
        self.line1 = nn.Linear(hidden_dim * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.ligand1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.receptor1 = nn.Linear(hidden_dim, hidden_dim * 4)
        
        self.ligand2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.receptor2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, ligand_batch, receptor_batch):
        ligand_out_seq, ligand_out_graph, ligand_mask_seq, ligand_mask_graph, ligand_seq_final, ligand_graph_final = self.ligand_graph_model(ligand_batch, device)
        receptor_out_seq, receptor_out_graph, receptor_mask_seq, receptor_mask_graph, receptor_seq_final, receptor_graph_final = self.receptor_graph_model(receptor_batch, device)
        
        context_layer, attention_score = self.cross_attention(
            [ligand_out_seq, ligand_out_graph, receptor_out_seq, receptor_out_graph],
            [ligand_mask_seq, ligand_mask_graph, receptor_mask_seq, receptor_mask_graph],
            device
        )

        out_ligand = context_layer[-1][0]  # Shape: (batch_size, 2 * max_nodes, 128)
        out_receptor = context_layer[-1][1]  # Shape: (batch_size, 2 * max_nodes, 128)
        
        # Concatenate masks to match out_ligand's node dimension
        ligand_mask_combined = torch.cat((ligand_mask_seq, ligand_mask_graph), dim=1)  # Shape: (batch_size, 2 * max_nodes)
        receptor_mask_combined = torch.cat((receptor_mask_seq, receptor_mask_graph), dim=1)  # Shape: (batch_size, 2 * max_nodes)
        
        # Affinity Prediction Module
        ligand_cross_seq = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_seq_final) / 2
        ligand_cross_stru = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_graph_final) / 2        

        ligand_cross = (ligand_cross_seq + ligand_cross_stru) / 2
        ligand_cross = self.ligand2(self.dropout(self.relu(self.ligand1(ligand_cross))))

        receptor_cross_seq = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_seq_final) / 2
        receptor_cross_stru = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_graph_final) / 2
        
        receptor_cross = (receptor_cross_seq + receptor_cross_stru) / 2
        receptor_cross = self.receptor2(self.dropout(self.relu(self.receptor1(receptor_cross))))   
        
        out = torch.cat((ligand_cross, receptor_cross), 1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        
        return out
# Initialize model
model = PPI().to(device)

# Load and rename state dictionary
state_dict = torch.load("save/weights/model_cv_(updated)2_2_1.pth", map_location=torch.device(device))
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("rna", "ligand").replace("mole", "receptor")
    new_state_dict[new_key] = value

# Load the renamed state dictionary
model.load_state_dict(new_state_dict)
model.eval()

# Input sequences
ligand = "MGDKPIWEQIGSSFIQHYYQLFDNDRTQLGAIYIDASCLTWEGQQFQGKAAIVEKLSSLPFQKIQHSITAQDHQPTPDSCIISMVVGQLKADEDPIMGFHQMFLLKNINDAWVCTNDMFRLALHNFG"
receptor = "MAAQGEPQVQFKLVLVGDGGTGKTTFVKRHLTGEFEKKYVATLGVEVHPLVFHTNRGPIKFNVWDTAGQEKFGGLRDGYYIQAQCAIIMFDVTSRVTYKNVPNWHRDLVRVCENIPIVLCGNKVDIKDRKVKAKSIVFHRKKNLQYYDISAKSNYNFEKPFLWLARKLIGDPNLEFVAMPALAPPEVVMDPALAAQYEHDLEVAQTTALPDEDDDL"

# Process sequences
process_ligand = ProteinInference(sequence=ligand)
processed_ligand = process_ligand.process()
process_receptor = ProteinInference(sequence=receptor)
processed_receptor = process_receptor.process()

# Print processed data (for debugging)
print(processed_ligand)
print(processed_receptor)

# Run inference
o = model(processed_ligand.to(device), processed_receptor.to(device)).item()
print(o)