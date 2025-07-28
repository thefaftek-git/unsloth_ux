






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
    from fine_tuning_framework import FineTuningFramework as GPUFrameworkClass
    GPU_AVAILABLE = True
except (NotImplementedError, ImportError):
    print("GPU framework unavailable, using CPU and mock frameworks")
    try:
        from mock_fine_tuning_framework import MockFineTuningFramework as MockFrameworkClass
        GPU_AVAILABLE = False
    except ImportError:
        print("Using mock framework only")
        from mock_fine_tuning_framework import MockFineTuningFramework as MockFrameworkClass
        GPU_AVAILABLE = False
    
# Function to choose appropriate framework based on model
def get_framework_class(model_name: str):
    """Choose the appropriate framework based on model type."""
    # Always use GPU framework if available
    if GPU_AVAILABLE:
        return GPUFrameworkClass
    else:
        return MockFrameworkClass

def get_hf_token():
    """Get HF token from file or environment."""
    import os
    
    # Try to read from hf_token file
    try:
        with open('hf_token', 'r') as f:
            token = f.read().strip()
            if token:
                return token
    except FileNotFoundError:
        pass
    
    # Try environment variables
    return os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

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
                # Check for different dataset formats
                parquet_files_root = list(item.glob("*.parquet"))
                jsonl_files_root = list(item.glob("*.jsonl")) + list(item.glob("data.jsonl"))
                json_files_root = list(item.glob("*.json"))
                arrow_files_root = list(item.glob("*.arrow"))
                dataset_info_file = item / "dataset_info.json"
                
                # Also check in data/ subdirectory
                data_dir = item / "data"
                if data_dir.exists() and data_dir.is_dir():
                    parquet_files_data = list(data_dir.glob("*.parquet"))
                    jsonl_files_data = list(data_dir.glob("*.jsonl"))
                    json_files_data = list(data_dir.glob("*.json"))
                    arrow_files_data = list(data_dir.glob("*.arrow"))
                else:
                    parquet_files_data = []
                    jsonl_files_data = []
                    json_files_data = []
                    arrow_files_data = []

                print(f"Found parquet files (root): {parquet_files_root}")
                print(f"Found parquet files (data/): {parquet_files_data}")
                print(f"Found jsonl files (root): {jsonl_files_root}")
                print(f"Found jsonl files (data/): {jsonl_files_data}")
                print(f"Found json files (root): {json_files_root}")
                print(f"Found json files (data/): {json_files_data}")
                print(f"Found arrow files (root): {arrow_files_root}")
                print(f"Found arrow files (data/): {arrow_files_data}")
                print(f"Found dataset_info.json: {dataset_info_file.exists()}")

                # Consider it a dataset if it has any of these file types or dataset_info.json
                if (parquet_files_root or jsonl_files_root or json_files_root or arrow_files_root or
                    parquet_files_data or jsonl_files_data or json_files_data or arrow_files_data or
                    dataset_info_file.exists()):
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

from pydantic import BaseModel
from typing import Optional

class LoadModelRequest(BaseModel):
    model_name: str = "gpt2"
    hf_token: Optional[str] = None
    full_finetuning: bool = False  # New parameter for full fine-tuning

class DownloadDatasetRequest(BaseModel):
    dataset_name: str
    subset_size: int = 1000

@app.post("/download-dataset/")
def download_dataset(request: DownloadDatasetRequest):
    """
    Download a dataset from Hugging Face and save it locally.

    Args:
        request (DownloadDatasetRequest): Request body containing dataset_name and subset_size.

    Returns:
        Dict[str, Any]: Status and dataset information.
    """
    try:
        from datasets import load_dataset
        from pathlib import Path

        # Create datasets directory if it doesn't exist
        datasets_dir = Path("./datasets")
        datasets_dir.mkdir(exist_ok=True)

        # Create a directory for this dataset
        dataset_dir_name = request.dataset_name.replace("/", "_")  # Replace slash for filesystem safety
        dataset_dir = datasets_dir / dataset_dir_name
        
        # Check if dataset already exists
        if dataset_dir.exists():
            return {
                "status": "success", 
                "message": f"Dataset {request.dataset_name} already exists locally at {dataset_dir}",
                "dataset_path": str(dataset_dir)
            }

        print(f"Downloading dataset {request.dataset_name}...")
        
        # Download the dataset from Hugging Face
        dataset = load_dataset(request.dataset_name, split='train')
        
        # Take subset if requested
        if request.subset_size and len(dataset) > request.subset_size:
            print(f"Taking subset of {request.subset_size} samples from {len(dataset)} total samples...")
            dataset = dataset.select(range(request.subset_size))

        # Create the dataset directory
        dataset_dir.mkdir(exist_ok=True)

        # Save the dataset in its original format
        print(f"Saving dataset to {dataset_dir}...")
        dataset.save_to_disk(str(dataset_dir))

        print(f"Successfully downloaded and saved {request.dataset_name} with {len(dataset)} samples")
        
        return {
            "status": "success",
            "message": f"Dataset {request.dataset_name} downloaded successfully",
            "dataset_path": str(dataset_dir),
            "num_samples": len(dataset)
        }

    except Exception as e:
        print(f"Error downloading dataset {request.dataset_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download dataset: {str(e)}")

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

            # Download FineTome-100k dataset locally in its original format
            finetome_dataset = load_dataset("mlabonne/FineTome-100k", split='train')

            # Create a directory for the default dataset
            default_dataset_dir = datasets_dir / "FineTome-100k"
            default_dataset_dir.mkdir(exist_ok=True)

            # Save the dataset in its original format (parquet)
            print(f"Saving FineTome-100k dataset to {default_dataset_dir}")
            
            # Save the dataset to disk maintaining its original structure
            finetome_dataset.save_to_disk(str(default_dataset_dir))

            print(f"Downloaded and saved FineTome-100k dataset with {len(finetome_dataset)} samples")
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

# Create a global framework instance - will be replaced when loading models
framework_instance = None

def validate_framework_instance():
    """Check if framework instance exists and raise error if not."""
    if framework_instance is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first using /load-model/ endpoint.")

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
def load_model(request: LoadModelRequest):
    """
    Load a pre-trained model for fine-tuning.

    Args:
        request (LoadModelRequest): Request body containing model_name, optional hf_token, and full_finetuning flag.

    Returns:
        Dict[str, Any]: Status information.
    """
    global framework_instance
    try:
        # Choose appropriate framework based on model type
        FrameworkClass = get_framework_class(request.model_name)
        framework_instance = FrameworkClass()
        
        # Use provided token or get from file/environment
        hf_token = request.hf_token or get_hf_token()
        
        framework_instance.load_model(request.model_name, hf_token=hf_token, full_finetuning=request.full_finetuning)
        
        training_mode = "full fine-tuning" if request.full_finetuning else "LoRA fine-tuning"
        return {
            "status": "success", 
            "model_loaded": request.model_name,
            "training_mode": training_mode
        }
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
        validate_framework_instance()
        
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
    If the path doesn't exist locally, attempt to download it from Hugging Face.

    Args:
        request (DatasetPathRequest): Request body containing dataset_path and subset_size.

    Returns:
        Dict[str, Any]: Status and dataset information.
    """
    global framework_instance

    if not request.dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path must be provided")

    try:
        validate_framework_instance()
        
        dataset_path = request.dataset_path
        
        # Check if this looks like a Hugging Face dataset name (contains slash and doesn't exist locally)
        if "/" in dataset_path and not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} looks like a Hugging Face dataset name. Attempting to download...")
            
            # Download the dataset first
            download_request = DownloadDatasetRequest(dataset_name=dataset_path, subset_size=request.subset_size)
            download_result = download_dataset(download_request)
            if download_result["status"] == "success":
                dataset_path = download_result["dataset_path"]
                print(f"Dataset downloaded to {dataset_path}")
            else:
                raise Exception(f"Failed to download dataset: {download_result}")

        # Load dataset using the provided path (now local)
        dataset = framework_instance.load_dataset(dataset_path, subset_size=request.subset_size)
        
        # Automatically preprocess the data
        framework_instance.preprocess_data()

        return {
            "status": "success",
            "dataset_path": dataset_path,
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
        validate_framework_instance()
        
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
        validate_framework_instance()
        
        # Setup training configuration
        framework_instance.setup_training(output_dir=request.output_dir, max_steps=request.max_steps)

        # Execute training
        stats = framework_instance.train()
        return {"status": "success", "training_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/setup-training/")
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
        validate_framework_instance()
        
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
        validate_framework_instance()
        
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
        validate_framework_instance()
        
        framework_instance.save_model(output_dir=output_dir)
        return {"status": "success", "model_saved_to": output_dir}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the app with CORS enabled and accessible from any host
    uvicorn.run(app, host="0.0.0.0", port=8071, log_level="info")






