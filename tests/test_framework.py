



"""
Test script for the fine-tuning framework.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Try to use the real framework if GPU is available
    from fine_tuning_framework import FineTuningFramework as FrameworkClass
except NotImplementedError:
    # Fall back to mock framework for CPU environments
    print("Using mock framework due to GPU requirement")
    from mock_fine_tuning_framework import MockFineTuningFramework as FrameworkClass

def test_framework():
    """Test the fine-tuning framework."""
    print("Testing fine-tuning framework...")

    # Create a temporary dataset file
    temp_dir = Path(tempfile.mkdtemp())
    dataset_file = temp_dir / "test_dataset.json"

    # Create a simple test dataset
    test_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."}
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What is 2 + 2?"},
                {"from": "gpt", "value": "2 + 2 equals 4."}
            ]
        }
    ]

    # Write test data to file
    import json
    with open(dataset_file, 'w') as f:
        json.dump(test_data, f)

    try:
        # Initialize framework
        framework = FrameworkClass()

        # Test model loading (skip due to GPU requirement)
        print("Testing model loading...")
        try:
            framework.load_model(model_name="unsloth/Pixtral-12B-Base-2409-bnb-4bit")
            assert framework.model is not None, "Model should be loaded"
            assert framework.processor is not None, "Processor should be loaded"
        except NotImplementedError as e:
            if "GPU" in str(e):
                print("Skipping model loading test due to GPU requirement")
            else:
                raise

        # Test dataset loading
        print("Testing dataset loading...")
        dataset = framework.load_dataset(str(dataset_file), subset_size=2)
        assert len(dataset) == 2, f"Dataset should have 2 samples, got {len(dataset)}"

        # Test data preprocessing
        print("Testing data preprocessing...")
        framework.preprocess_data()
        assert framework.dataset is not None, "Dataset should be preprocessed"

        # Test training setup (without actually running training)
        print("Testing training setup...")
        framework.setup_training(output_dir=str(temp_dir / "outputs"), max_steps=1)

        # Verify trainer was created
        assert framework.trainer is not None, "Trainer should be set up"
        if isinstance(framework.trainer, dict):
            assert 'model' in framework.trainer, "Trainer should have model config"

        print("All tests passed!")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = test_framework()
    sys.exit(0 if success else 1)


