from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound", example="CCO")

class BatchPredictionRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings")

class PredictionResponse(BaseModel):
    smiles: Optional[str] = None
    prediction: Optional[float] = None
    model: str
    error: Optional[str] = None