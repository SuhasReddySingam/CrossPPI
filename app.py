import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import Dict

# Import your model and processor
from model import Protein_feature_extraction, cross_attention
from data.protein_processor import ProteinInference
import torch.nn as nn

# ========================= CONFIG =========================
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "save/model_cv_(updated)2_2_1.pth"
HIDDEN_DIM = 128

# ========================= MODEL DEFINITION =========================
class PPI(nn.Module):
    def __init__(self):
        super(PPI, self).__init__()
        self.ligand_graph_model = Protein_feature_extraction(HIDDEN_DIM)
        self.receptor_graph_model = Protein_feature_extraction(HIDDEN_DIM)
        self.cross_attention = cross_attention(HIDDEN_DIM)

        self.line1 = nn.Linear(HIDDEN_DIM * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

        self.ligand1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4)
        self.receptor1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4)
        self.ligand2 = nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM)
        self.receptor2 = nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM)

        self.relu = nn.ReLU()

    def forward(self, ligand_batch, receptor_batch):
        ligand_out_seq, ligand_out_graph, ligand_mask_seq, ligand_mask_graph, ligand_seq_final, ligand_graph_final = \
            self.ligand_graph_model(ligand_batch, DEVICE)

        receptor_out_seq, receptor_out_graph, receptor_mask_seq, receptor_mask_graph, receptor_seq_final, receptor_graph_final = \
            self.receptor_graph_model(receptor_batch, DEVICE)

        context_layer, _ = self.cross_attention(
            [ligand_out_seq, ligand_out_graph, receptor_out_seq, receptor_out_graph],
            [ligand_mask_seq, ligand_mask_graph, receptor_mask_seq, receptor_mask_graph],
            DEVICE
        )

        out_ligand = context_layer[-1][0]
        out_receptor = context_layer[-1][1]

        ligand_mask_combined = torch.cat((ligand_mask_seq, ligand_mask_graph), dim=1)
        receptor_mask_combined = torch.cat((receptor_mask_seq, receptor_mask_graph), dim=1)

        # Ligand cross
        ligand_cross_seq = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_seq_final) / 2
        ligand_cross_stru = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_graph_final) / 2
        ligand_cross = (ligand_cross_seq + ligand_cross_stru) / 2
        ligand_cross = self.ligand2(self.dropout(self.relu(self.ligand1(ligand_cross))))

        # Receptor cross
        receptor_cross_seq = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_seq_final) / 2
        receptor_cross_stru = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_graph_final) / 2
        receptor_cross = (receptor_cross_seq + receptor_cross_stru) / 2
        receptor_cross = self.receptor2(self.dropout(self.relu(self.receptor1(receptor_cross))))

        out = torch.cat((ligand_cross, receptor_cross), 1)
        out = self.dropout(self.relu(self.line1(out)))
        out = self.dropout(self.relu(self.line2(out)))
        out = self.line3(out)

        return out


# ========================= LOAD MODEL =========================
model = PPI().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Rename keys if needed (rna -> ligand, mole -> receptor)
new_state_dict = {k.replace("rna", "ligand").replace("mole", "receptor"): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

print(f"Model loaded successfully on {DEVICE}")

# ========================= FASTAPI =========================
app = FastAPI(title="Protein-Protein Interaction Affinity Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # For development - allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ligand: str
    receptor: str


class PredictionResponse(BaseModel):
    predicted_affinity: float
    success: bool


@app.post("/predict", response_model=PredictionResponse)
async def predict_affinity(request: PredictionRequest):
    try:
        # Process sequences
        process_ligand = ProteinInference(sequence=request.ligand)
        processed_ligand = process_ligand.process()

        process_receptor = ProteinInference(sequence=request.receptor)
        processed_receptor = process_receptor.process()

        # Move to device
        # Assuming your ProteinInference returns a PyG Data object or dict
        if hasattr(processed_ligand, 'to'):
            ligand_batch = processed_ligand.to(DEVICE)
            receptor_batch = processed_receptor.to(DEVICE)
        else:
            # If it's already a dict/tuple, wrap accordingly
            ligand_batch = processed_ligand
            receptor_batch = processed_receptor

        # Inference
        with torch.no_grad():
            output = model(ligand_batch, receptor_batch)
            affinity = output.item()

        return PredictionResponse(
            predicted_affinity=round(affinity, 6),
            success=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(DEVICE)}


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)