




# Fine-Tuning Framework with Web UI

A clean, reusable framework for fine-tuning transformer models with CPU support and a web interface.

## Features

- **CPU Compatible**: Full fine-tuning workflow on CPU
- **Web Interface**: Easy-to-use web UI for selecting datasets and starting jobs
- **Modular Design**: Clean separation of concerns
- **Tested**: Comprehensive unit tests and validation
- **Dataset Support**: Works with FineTome-100k dataset

## Project Structure

```
fine_tuning_project/
├── src/                  # Core framework implementation
│   ├── fine_tuning_framework_cpu.py  # CPU-compatible fine-tuning framework
│   └── mock_fine_tuning_framework.py # Mock for testing
├── tests/                # Unit tests
│   └── test_cpu_finetuning_with_finetome.py
├── web/                  # Web interface
│   ├── app.py            # FastAPI backend
│   └── frontend.html     # HTML frontend
├── venv/                 # Virtual environment
└── requirements.txt      # Dependencies
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fine_tuning_project
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### API Endpoints

The web interface provides the following endpoints:

- `GET /`: Root endpoint
- `POST /load-model/?model_name=gpt2`: Load a pre-trained model
- `POST /load-dataset/?dataset_path=/path/to/dataset&subset_size=100`: Load dataset for fine-tuning
- `POST /setup-training/?output_dir=outputs&max_steps=30`: Set up training configuration
- `POST /train/`: Execute fine-tuning training
- `POST /save-model/?output_dir=fine_tuned_model`: Save the fine-tuned model

### Web Interface

1. **Start the web server**:
   ```bash
   python web/app.py
   ```

2. **Open the frontend** in your browser:
   [http://localhost:8020/frontend.html](http://localhost:8020/frontend.html)

3. **Use the interface**:
   - Select a model (GPT-2 or DistilBERT)
   - Choose a dataset from available options
   - Configure training parameters
   - Start fine-tuning and save the model

## Testing

Run unit tests with:
```bash
python -m pytest tests/test_cpu_finetuning_with_finetome.py -v
```

## Implementation Details

- **CPU Support**: Uses transformers library with CPU device specification
- **Dataset Handling**: Supports loading from existing directories or file uploads
- **Error Handling**: Comprehensive error handling throughout the workflow
- **Progress Tracking**: Real-time feedback during training operations

## Dependencies

See `requirements.txt` for all dependencies. Key libraries include:
- transformers
- datasets
- fastapi
- torch

## License

This project is licensed under the MIT License.




