






"""
Test the CPU fine-tuning framework with actual training using FineTome dataset.
"""

import os
import sys
sys.path.append('/workspace/fine_tuning_project/src')
from fine_tuning_framework_cpu import FineTuningFramework

def test_cpu_finetuning_with_finetome():
    """Test complete fine-tuning workflow on CPU with FineTome dataset."""
    print("Testing CPU fine-tuning framework with FineTome dataset...")

    # Initialize framework
    framework = FineTuningFramework()

    # Load model (use a small model for CPU)
    framework.load_model("gpt2")

    # Load FineTome dataset (10 samples for quick testing)
    finetome_path = "/workspace/FineTome-100k"
    print(f"Loading FineTome dataset from {finetome_path}...")
    framework.load_dataset(finetome_path, subset_size=10)

    # Preprocess data
    framework.preprocess_data()

    # Set up training (very small number of steps for quick test)
    framework.setup_training(output_dir="cpu_test_outputs", max_steps=5)

    # Run training
    stats = framework.train()

    # Save model
    framework.save_model("cpu_fine_tuned_model")

    print(f"Training completed. Runtime: {stats.metrics['train_runtime']:.2f}s")
    return True

if __name__ == "__main__":
    success = test_cpu_finetuning_with_finetome()
    if success:
        print("CPU fine-tuning with FineTome test passed!")
    else:
        print("CPU fine-tuning with FineTome test failed!")



