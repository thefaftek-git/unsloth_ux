






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
    # Mount frontend at /web endpoint
    app.mount("/web", StaticFiles(directory=static_dir), name="frontend")
    # Also serve index.html at root for convenience
    @app.get("/")
    def get_index():
        return HTMLResponse(open(static_dir / "index.html").read())

@app.get("/api")
def read_api_root():
    """API root endpoint."""
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

@app.get("/huggingface-search-datasets/")
def search_huggingface_datasets(query: str = "", limit: int = 10):
    """
    Search for datasets on Hugging Face hub.

    Args:
        query (str): Search query. Defaults to "" (returns popular datasets).
        limit (int): Number of results to return. Defaults to 10.

    Returns:
        Dict[str, Any]: Status and list of dataset information.
    """
    try:
        import requests

        # Base URL for Hugging Face datasets API
        api_url = "https://huggingface.co/api/datasets"

        # Prepare query parameters
        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",  # Sort by popularity
            "direction": -1,     # Descending order
            "tags": "public"      # Only public datasets
        }

        # Make request to Hugging Face API
        response = requests.get(api_url, params=params)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Hugging Face API error: {response.text}")

        data = response.json()

        # Extract relevant information for each dataset
        datasets = []
        for item in data:
            dataset_id = item.get("id", "")
            dataset_info = {
                "id": dataset_id,
                "name": dataset_id.split("/")[-1],
                "full_name": dataset_id,
                "description": item.get("tags", []),
                "downloads": item.get("downloads", 0)
            }
            datasets.append(dataset_info)

        return {
            "status": "success",
            "datasets": datasets,
            "count": len(datasets),
            "query": query
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function to check and download default dataset if needed
def ensure_default_dataset():
    """Check if datasets directory is empty and download FineTome-100k if so."""
    import requests
    from pathlib import Path

    datasets_dir = Path("./datasets")
    print(f"Checking datasets directory: {datasets_dir}")

    # Create datasets directory if it doesn't exist
    datasets_dir.mkdir(exist_ok=True)

    # Check if directory is empty (no dataset folders)
    dataset_folders = [d for d in datasets_dir.iterdir() if d.is_dir()]
    print(f"Found existing datasets: {len(dataset_folders)}")

    if len(dataset_folders) == 0:
        print("No datasets found, attempting to download FineTome-100k...")

        # Try to load the dataset from Hugging Face
        try:
            from datasets import load_dataset

            # Download and save FineTome-100k dataset locally
            finetome_dataset = load_dataset("mlabonne/FineTome-100k", split='train')

            # Create a directory for the default dataset
            default_dataset_dir = datasets_dir / "FineTome-100k"
            default_dataset_dir.mkdir(exist_ok=True)

            # Save as JSONL format (compatible with our framework)
            save_path = default_dataset_dir / "data.jsonl"

            print(f"Saving FineTome-100k dataset to {save_path}")

            # Convert to JSON lines format
            with open(save_path, 'w', encoding='utf-8') as f:
                for i, example in enumerate(finetome_dataset):
                    if i >= 100:  # Limit to first 100 samples for default dataset
                        break
                    json_str = {
                        "conversations": [
                            {"from": "user", "value": example.get("text", "")}
                        ]
                    }
                    f.write(f"{json_str}\n")

            print(f"Downloaded and saved FineTome-100k default dataset with {i+1} samples")
            return True

        except Exception as e:
            print(f"Failed to download default dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Datasets found, skipping default dataset download")
        return True

# Check for default dataset at startup
ensure_default_dataset()

# Create a global framework instance
framework_instance = FrameworkClass()

@app.get("/huggingface-search-models/")
def search_huggingface_models(query: str = "", limit: int = 10):
    """
    Search for models on Hugging Face hub.

    Args:
        query (str): Search query. Defaults to "" (returns popular models).
        limit (int): Number of results to return. Defaults to 10.

    Returns:
        Dict[str, Any]: Status and list of model information.
    """
    try:
        import requests

        # Base URL for Hugging Face models API
        api_url = "https://huggingface.co/api/models"

        # Prepare query parameters
        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",  # Sort by popularity
            "direction": -1,     # Descending order
            "tags": "public"      # Only public models
        }

        # Make request to Hugging Face API
        response = requests.get(api_url, params=params)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Hugging Face API error: {response.text}")

        data = response.json()

        # Extract relevant information for each model
        models = []
        for item in data:
            model_id = item.get("id", "")
            model_info = {
                "id": model_id,
                "name": model_id.split("/")[-1],
                "full_name": model_id,
                "description": item.get("tags", []),
                "downloads": item.get("downloads", 0)
            }
            models.append(model_info)

        return {
            "status": "success",
            "models": models,
            "count": len(models),
            "query": query
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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






