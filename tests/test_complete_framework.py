







"""
Complete end-to-end test of the fine-tuning framework.
"""

import os
import sys
sys.path.append('/workspace/fine_tuning_project/src')
from fine_tuning_framework_cpu import FineTuningFramework

def test_complete_finetuning_workflow():
    """Test complete fine-tuning workflow from start to finish."""
    print("Testing complete fine-tuning workflow...")

    # Initialize framework
    framework = FineTuningFramework()

    # Load model (use a small model for CPU)
    framework.load_model("gpt2")

    # Load dataset (10 samples for quick testing)
    finetome_path = "/workspace/FineTome-100k"
    print(f"Loading FineTome dataset from {finetome_path}...")
    framework.load_dataset(finetome_path, subset_size=10)

    # Preprocess data
    framework.preprocess_data()

    # Set up training (small number of steps for quick test)
    framework.setup_training(output_dir="test_outputs", max_steps=5)

    # Run training
    stats = framework.train()

    # Save model
    framework.save_model("test_fine_tuned_model")

    print(f"Complete workflow completed successfully!")
    print(f"Training runtime: {stats.metrics['train_runtime']:.2f}s")
    print(f"Final loss: {stats.metrics.get('loss', 'N/A')}")

    # Verify model files were created
    assert os.path.exists("test_fine_tuned_model/config.json"), "Model config file not found"
    assert os.path.exists("test_fine_tuned_model/model.safetensors") or os.path.exists("test_fine_tuned_model/pytorch_model.bin"), "Model weights file not found"

    return True

if __name__ == "__main__":
    success = test_complete_finetuning_workflow()
    if success:
        print("\n✅ Complete fine-tuning workflow test PASSED!")
    else:
        print("\n❌ Complete fine-tuning workflow test FAILED!")




