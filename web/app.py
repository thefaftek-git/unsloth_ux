






"""
Web UI for fine-tuning framework using FastAPI.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from fine_tuning_framework_cpu import FineTuningFramework as FrameworkClass
except NotImplementedError:
    print("Using mock framework due to GPU requirement")
    from mock_fine_tuning_framework import MockFineTuningFramework as FrameworkClass

app = FastAPI()

# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the "frontend" directory
static_dir = Path(__file__).parent / "frontend"
if static_dir.exists():
    app.mount("/web", StaticFiles(directory=static_dir), name="frontend")

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Fine-tuning framework API"}

@app.get("/available-datasets/")
def get_available_datasets(base_dir: str = "./datasets"):
    """
    Get a list of available datasets in the specified directory.

    Args:
        base_dir (str): Base directory to search for datasets. Defaults to "./datasets".

    Returns:
        Dict[str, List[str]]: Status and list of dataset paths.
    """
    try:
        # Scan the base directory for potential dataset directories
        base_path = Path(base_dir)
        if not base_path.exists():
            return {"status": "error", "message": f"Base directory {base_dir} does not exist"}

        datasets = []
        print(f"Scanning directory: {base_path}")
        for item in base_path.iterdir():
            print(f"Checking item: {item}")
            if item.is_dir():
                # Check both root and data subdirectory
                parquet_files_root = list(item.glob("*.parquet"))
                jsonl_files_root = list(item.glob("data.jsonl"))

                # Also check in data/ subdirectory
                data_dir = item / "data"
                if data_dir.exists() and data_dir.is_dir():
                    parquet_files_data = list(data_dir.glob("*.parquet"))
                    jsonl_files_data = list(data_dir.glob("data.jsonl"))
                else:
                    parquet_files_data = []
                    jsonl_files_data = []

                print(f"Found parquet files (root): {parquet_files_root}")
                print(f"Found parquet files (data/): {parquet_files_data}")
                print(f"Found jsonl files (root): {jsonl_files_root}")
                print(f"Found jsonl files (data/): {jsonl_files_data}")

                if parquet_files_root or jsonl_files_root or parquet_files_data or jsonl_files_data:
                    # Check for common dataset file patterns (parquet or jsonl)
                    datasets.append(str(item))
                    print(f"Added dataset: {item}")

        return {
            "status": "success",
            "datasets": datasets,
            "count": len(datasets)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Create a global framework instance
framework_instance = FrameworkClass()

@app.post("/load-model/")
def load_model(model_name: str = "gpt2"):
    """
    Load a pre-trained model for fine-tuning.

    Args:
        model_name (str): Name of the model to load. Defaults to "gpt2".

    Returns:
        Dict[str, Any]: Status information.
    """
    global framework_instance
    try:
        # Only reset if we're changing models or starting fresh
        if framework_instance.model is None or framework_instance.tokenizer is None:
            framework_instance = FrameworkClass()
        framework_instance.load_model(model_name)
        return {"status": "success", "model_loaded": model_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/load-dataset/")
async def load_dataset(file: UploadFile = File(None), dataset_path: str = None, subset_size: int = 100):
    """
    Load a dataset for fine-tuning.

    Args:
        file (UploadFile): Dataset file to upload. Optional if dataset_path is provided.
        dataset_path (str): Path to existing dataset directory. Optional if file is uploaded.
        subset_size (int): Number of samples to use. Defaults to 100.

    Returns:
        Dict[str, Any]: Status and dataset information.
    """
    global framework_instance

    # Validate that either file or dataset_path is provided
    if not file and not dataset_path:
        raise HTTPException(status_code=400, detail="Either 'file' or 'dataset_path' must be provided")

    try:
        if file:
            # Save uploaded file temporarily
            temp_dir = Path("/tmp/fine_tuning")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / file.filename

            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)

            # Load dataset from uploaded file
            dataset_path = str(temp_file)

        # Load dataset using the provided path or uploaded file
        dataset = framework_instance.load_dataset(dataset_path, subset_size=subset_size)

        try:
            # Preprocess the data immediately after loading
            framework_instance.preprocess_data()
            return {
                "status": "success",
                "dataset_path": dataset_path,
                "num_samples": len(dataset)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file if it was created
        if 'temp_file' in locals() and temp_file.exists():
            os.unlink(str(temp_file))

from pydantic import BaseModel

class DatasetPathRequest(BaseModel):
    dataset_path: str
    subset_size: int = 100

@app.post("/load-dataset-from-path/")
def load_dataset_from_path(request: DatasetPathRequest):
    """
    Load a dataset for fine-tuning from a specified path.

    Args:
        request (DatasetPathRequest): Request body containing dataset_path and subset_size.

    Returns:
        Dict[str, Any]: Status and dataset information.
    """
    global framework_instance

    if not request.dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path must be provided")

    try:
        # Load dataset using the provided path and preprocess it in one step
        dataset = framework_instance.load_dataset(request.dataset_path, subset_size=request.subset_size)

        return {
            "status": "success",
            "dataset_path": request.dataset_path,
            "num_samples": len(dataset)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/preprocess-data/")
def preprocess_data():
    """
    Preprocess the dataset for fine-tuning.

    Returns:
        Dict[str, Any]: Status information.
    """
    global framework_instance
    try:
        framework_instance.preprocess_data()
        return {"status": "success", "message": "Data preprocessing completed"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from pydantic import BaseModel

class TrainingRequest(BaseModel):
    output_dir: str = "outputs"
    max_steps: int = 30

@app.post("/train-model/")
def train_model(request: TrainingRequest):
    """
    Set up and execute the fine-tuning training in one step.

    Args:
        request (TrainingRequest): Request body containing output_dir and max_steps.

    Returns:
        Dict[str, Any]: Training statistics.
    """
    global framework_instance
    try:
        # Setup training configuration
        framework_instance.setup_training(output_dir=request.output_dir, max_steps=request.max_steps)

        # Execute training
        stats = framework_instance.train()
        return {"status": "success", "training_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/setup-training/")
def setup_training(output_dir: str = "outputs", max_steps: int = 30):
    """
    Set up the training configuration.

    Args:
        output_dir (str): Directory to save training outputs.
        max_steps (int): Number of training steps. Defaults to 30.

    Returns:
        Dict[str, Any]: Status information.
    """
    global framework_instance
    try:
        framework_instance.setup_training(output_dir=output_dir, max_steps=max_steps)
        return {"status": "success", "message": "Training configuration set up"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/")
def train():
    """
    Execute the fine-tuning training.

    Returns:
        Dict[str, Any]: Training statistics.
    """
    global framework_instance
    try:
        stats = framework_instance.train()
        return {"status": "success", "training_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save-model/")
def save_model(output_dir: str = "fine_tuned_model"):
    """
    Save the fine-tuned model.

    Args:
        output_dir (str): Directory to save the model.

    Returns:
        Dict[str, Any]: Status information.
    """
    global framework_instance
    try:
        framework_instance.save_model(output_dir=output_dir)
        return {"status": "success", "model_saved_to": output_dir}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the app with CORS enabled and accessible from any host
    uvicorn.run(app, host="0.0.0.0", port=8070, log_level="info")






