





"""
Fine-tuning framework using transformers libraries for CPU environments.
This module provides a clean, reusable interface for fine-tuning models without GPU requirements.
"""

import os
from typing import Optional, Dict, Any, List

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

class FineTuningFramework:
    """
    A framework for fine-tuning models using the transformers library.
    This version is designed to work on CPU with smaller models.
    """

    def __init__(self):
        """Initialize the fine-tuning framework."""
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None

    def load_model(self, model_name: str = "gpt2"):
        """
        Load a pre-trained model for fine-tuning.

        Args:
            model_name (str): Name of the model to load. Defaults to "gpt2".
        """
        print(f"Loading model {model_name}...")
        try:
            # Check if we have GPU access
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            # Load the tokenizer and model using transformers directly
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Use float32 for CPU compatibility
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
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
            # Load local dataset - check file extension
            if dataset_path.endswith('.json') or (os.path.isdir(dataset_path) and any(f.endswith('.json') for f in os.listdir(dataset_path))):
                self.dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith('.parquet') or (os.path.isdir(dataset_path) and any(f.endswith('.parquet') for f in os.listdir(dataset_path))):
                # Handle parquet files
                if os.path.isdir(dataset_path):
                    # Find all parquet files in the directory
                    parquet_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.parquet')]
                    self.dataset = load_dataset('parquet', data_files=parquet_files)
                else:
                    self.dataset = load_dataset('parquet', data_files=dataset_path)
            else:
                # Try to auto-detect format
                try:
                    self.dataset = load_dataset(dataset_path, split='train')
                except Exception as e:
                    raise ValueError(f"Could not load dataset from {dataset_path}: {e}")
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

        # Convert dataset to the format expected by transformers
        def extract_conversations(example):
            """Extract conversations from a single example."""
            conversations = example.get('conversations', [])
            if not conversations:
                return ""

            # Extract all conversation content into a single string
            texts = []
            for conv in conversations:
                role = conv.get('from', 'user')
                content = conv.get('value', '')
                if content.strip():
                    texts.append(f"{role}: {content}")

            return " ".join(texts)

        # Apply conversion to all samples and tokenize
        print("Preprocessing data...")
        def tokenize_function(example):
            text = extract_conversations(example)
            return self.tokenizer(text, truncation=True, padding="max_length", max_length=128)

        self.dataset = dataset.map(tokenize_function, batched=False)
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

        # Configure trainer with CPU-friendly settings for causal LM
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Small batch size for CPU
            gradient_accumulation_steps=4,
            max_steps=max_steps,
            learning_rate=2e-5,  # Smaller LR for stability on CPU
            logging_steps=1,
            save_strategy="steps",
            optim="adamw_torch",  # Use standard AdamW for CPU
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to="none"
        )

        # Create data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Set to False for causal LM
            pad_to_multiple_of=8
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator
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
        self.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully.")



