from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse, BatchPredictionRequest
from app.services.engine import engine 

import pandas as pd
import io
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(data: PredictionRequest):
    """
    Predict saturation vapor pressure for a single SMILES string.
    """
    # Unpack the tuple returned by engine.predict_single
    prediction, model, error = engine.predict_single(data.smiles)

    if error:
        return PredictionResponse(model="Error", error=error)

    return PredictionResponse(
        smiles=data.smiles,
        prediction=prediction,
        model=model
    )

@router.post("/batch_predict")
async def batch_predict(request: Request):
    """
    Polymorphic endpoint handling both JSON batch lists and CSV file uploads
    to match the legacy frontend behavior.
    """
    content_type = request.headers.get("content-type", "")
    smiles_list = []
    is_file_request  = False

    # CASE A: JSON Batch Input
    if "application/json" in content_type:
        try:
            body = await request.json()
            smiles_list = body.get("smiles_list", [])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # CASE B: File Upload (Multipart)
    elif "multipart/form-data" in content_type:
        is_file_request = True
        form = await request.form()
        file = form.get("file")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file found in form data")

        # Read file
        contents = await file.read()
        try:
            # Assuming file is simple text with one SMILES per line or CSV
            decoded = contents.decode('utf-8')
            smiles_list = [line.strip() for line in decoded.splitlines() if line.strip()]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Use application/json or multipart/form-data")

    # --- Process Batch via Engine ---
    if not smiles_list:
        return [] if not is_file_request else JSONResponse([])

    results = engine.predict_batch(smiles_list)
    # --- Return Response ---
    if is_file_request:
        # Create CSV in memory for file download
        csv_data = []
        for res in results:
            pred = res['prediction']
            # Formatting logic
            rounded_pred = f"{float(pred):.2f}" if pred is not None else "N/A"
            log_pred = float(np.log10(float(pred))) if pred is not None else "N/A"
            
            csv_data.append({
                "SMILES": res['smiles'],
                "Prediction[Pa]": rounded_pred,
                "log10-Prediction[Pa]": log_pred,
                "Model": res['model']
            })
            
        df = pd.DataFrame(csv_data)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response
    else:
            # Return JSON
            clean_results = []
            for res in results:
            
                clean_results.append({
                    "smiles": res.get("smiles"),
                    # Explicitly cast to float
                    "predicted_vapor_pressure": float(res["prediction"]) if res.get("prediction") is not None else None,
                    "uncertainty": float(res["uncertainty"]) if res.get("uncertainty") is not None else None,
                    "model_version": res.get("model"),
                    "error": res.get("error") 
                })
                
            return clean_results