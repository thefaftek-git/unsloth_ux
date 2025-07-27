





"""
Test the CPU fine-tuning framework with actual training.
"""

import os
import sys
sys.path.append('/workspace/fine_tuning_project/src')
from fine_tuning_framework_cpu import FineTuningFramework

def test_cpu_finetuning():
    """Test complete fine-tuning workflow on CPU."""
    print("Testing CPU fine-tuning framework...")

    # Initialize framework
    framework = FineTuningFramework()

    # Load model (use a small model for CPU)
    framework.load_model("gpt2")

    # Create a simple test dataset
    test_data = {
        "conversations": [
            {"from": "user", "value": "Hello, how are you?"},
            {"from": "assistant", "value": "I'm fine, thank you!"}
        ]
    }

    import json
    # Use absolute path for dataset
    test_dataset_dir = "/workspace/fine_tuning_project/test_dataset"
    os.makedirs(test_dataset_dir, exist_ok=True)
    with open(os.path.join(test_dataset_dir, "test.json"), "w") as f:
        json.dump(test_data, f)

    # Load dataset (2 samples for quick testing)
    framework.load_dataset(test_dataset_dir, subset_size=2)

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
    success = test_cpu_finetuning()
    if success:
        print("CPU fine-tuning test passed!")
    else:
        print("CPU fine-tuning test failed!")


