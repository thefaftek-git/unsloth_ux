
"""
Fine-tuning framework using unsloth and transformers libraries.
This module provides a clean, reusable interface for fine-tuning models.
"""

import os
import warnings
from typing import Optional, Dict, Any, List

# Import unsloth first to ensure optimizations are applied
import unsloth

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

class FineTuningFramework:
    """
    A framework for fine-tuning models using the unsloth library.
    """

    def __init__(self):
        """Initialize the fine-tuning framework."""
        self.model = None
        self.processor = None
        self.trainer = None
        self.dataset = None

    def load_model(self, model_name: str = "gemma3n-4b"):
        """
        Load a pre-trained model for fine-tuning.

        Args:
            model_name (str): Name of the model to load. Defaults to "gemma3n-4b".

        Raises:
            NotImplementedError: If GPU is not available.
        """
        print(f"Loading model {model_name}...")
        try:
            self.model, self.processor = FastVisionModel.from_pretrained(model_name)
            print("Model loaded successfully.")
        except NotImplementedError as e:
            if "GPU" in str(e):
                error_msg = f"GPU is required for unsloth but not available. Error: {e}"
                warnings.warn(error_msg)
                raise NotImplementedError(f"No GPU available for model loading: {error_msg}") from e
            else:
                raise

    def load_dataset(self, dataset_path: str, subset_size: Optional[int] = 100):
        """
        Load a dataset for fine-tuning.

        Args:
            dataset_path (str): Path to the dataset.
            subset_size (Optional[int]): Number of samples to use. Defaults to 100.

        Returns:
            Dataset: The loaded dataset.
        """
        print(f"Loading dataset from {dataset_path}...")
        if os.path.exists(dataset_path):
            # Load local dataset
            self.dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            # Try to load from HuggingFace hub
            try:
                self.dataset = load_dataset(dataset_path, split='train')
            except Exception as e:
                raise ValueError(f"Could not load dataset from {dataset_path}: {e}")

        if subset_size and len(self.dataset) > subset_size:
            print(f"Using subset of {subset_size} samples...")
            self.dataset = self.dataset.select(range(subset_size))

        print(f"Dataset loaded with {len(self.dataset)} samples.")
        return self.dataset

    def preprocess_data(self, dataset: Optional[Dataset] = None):
        """
        Preprocess the dataset for fine-tuning.

        Args:
            dataset (Optional[Dataset]): Dataset to preprocess. Uses self.dataset if not provided.
        """
        if dataset is None:
            dataset = self.dataset

        if dataset is None:
            raise ValueError("No dataset available. Please load a dataset first.")

        # Convert dataset to the format expected by unsloth
        def convert_to_conversation(sample):
            """Convert sample to conversation format."""
            conversations = sample.get('conversations', [])
            if not conversations:
                return None

            # Convert to the format expected by the model
            converted_conversation = {
                "messages": []
            }

            for conv in conversations:
                role = conv.get('from', 'user')
                content = conv.get('value', '')
                message = {"role": role, "content": [{"type": "text", "text": content}]}
                converted_conversation["messages"].append(message)

            return converted_conversation

        # Apply conversion to all samples
        print("Preprocessing data...")
        self.dataset = dataset.map(lambda x: convert_to_conversation(x), remove_columns=dataset.column_names)
        print("Data preprocessing completed.")

    def setup_training(self, output_dir: str = "outputs", max_steps: int = 30):
        """
        Set up the training configuration.

        Args:
            output_dir (str): Directory to save training outputs.
            max_steps (int): Number of training steps. Defaults to 30.
        """
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be loaded before setting up training.")

        print("Setting up training configuration...")
        FastVisionModel.for_training(self.model)

        # Configure trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            processing_class=self.processor.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.processor),
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                max_steps=max_steps,
                learning_rate=2e-4,
                logging_steps=1,
                save_strategy="steps",
                optim="adamw_torch_fused",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir=output_dir,
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=2048,
            )
        )
        print("Training configuration set up successfully.")

    def train(self):
        """
        Execute the fine-tuning training.

        Returns:
            Dict[str, Any]: Training statistics.
        """
        if self.trainer is None:
            raise ValueError("Training must be set up before running.")

        print("Starting training...")
        trainer_stats = self.trainer.train()

        # Print training summary
        print(f"\nTraining completed in {trainer_stats.metrics['train_runtime']:.2f} seconds.")
        print(f"Peak memory usage: {trainer_stats.metrics.get('memory_peak', 'N/A')}")

        return trainer_stats

    def save_model(self, output_dir: str = "fine_tuned_model"):
        """
        Save the fine-tuned model.

        Args:
            output_dir (str): Directory to save the model.
        """
        if self.model is None:
            raise ValueError("No model available to save.")

        print(f"Saving model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print("Model saved successfully.")
