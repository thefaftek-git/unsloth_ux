



"""
Test script for loading and processing the FineTome-100k dataset.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fine_tuning_framework import FineTuningFramework as FrameworkClass
except NotImplementedError:
    print("Using mock framework due to GPU requirement")
    from mock_fine_tuning_framework import MockFineTuningFramework as FrameworkClass

def test_finetome_dataset():
    """Test loading and processing the FineTome-100k dataset."""
    print("Testing FineTome-100k dataset...")

    # Check if FineTome-100k is available
    finetome_path = "/workspace/FineTome-100k"
    print(f"Checking for FineTome-100k dataset at {finetome_path}")
    if not os.path.exists(finetome_path):
        print(f"FineTome-100k dataset not found at {finetome_path}")
        return False

    framework = FrameworkClass()

    try:
        # Load a subset of the FineTome-100k dataset (first 100 samples)
        dataset = framework.load_dataset(finetome_path, subset_size=100)

        print(f"Loaded {len(dataset)} samples from FineTome-100k")
        assert len(dataset) == 100, f"Expected 100 samples, got {len(dataset)}"

        # Test data preprocessing
        framework.preprocess_data()
        assert framework.dataset is not None, "Dataset should be preprocessed"

        print("FineTome-100k dataset test passed!")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_finetome_dataset()
    sys.exit(0 if success else 1)



