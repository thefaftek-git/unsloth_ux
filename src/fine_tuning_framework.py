
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
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class FineTuningFramework:
    """
    A framework for fine-tuning models using the unsloth library.
    """

    def __init__(self):
        """Initialize the fine-tuning framework."""
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.full_finetuning = False  # Track fine-tuning mode

    def load_model(self, model_name: str = "google/gemma-2-2b-it", hf_token: str = None, full_finetuning: bool = False):
        """
        Load a pre-trained model for fine-tuning.

        Args:
            model_name (str): Name of the model to load. Defaults to "google/gemma-2-2b-it".
            hf_token (str): Hugging Face token for accessing private models. Optional.
            full_finetuning (bool): Whether to enable full fine-tuning instead of LoRA. Defaults to False.

        Raises:
            NotImplementedError: If GPU is not available.
        """
        print(f"Loading model {model_name}...")
        
        # Store full_finetuning preference for later use
        self.full_finetuning = full_finetuning
        if full_finetuning:
            print("Full fine-tuning mode enabled - all model parameters will be trainable")
        else:
            print("LoRA fine-tuning mode enabled - only adapter parameters will be trainable")
        
        try:
            # Get token from environment if not provided
            if hf_token is None:
                import os
                hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
                if not hf_token:
                    # Try reading from hf_token file
                    try:
                        with open('hf_token', 'r') as f:
                            hf_token = f.read().strip()
                    except FileNotFoundError:
                        pass
            
            # Prepare kwargs for model loading
            model_kwargs = {
                'max_seq_length': 2048,
                'dtype': None,
                'load_in_4bit': not full_finetuning,  # Disable 4bit for full fine-tuning
                'load_in_8bit': False,
                'full_finetuning': full_finetuning,  # Enable full fine-tuning if requested
                'device_map': "auto" if not full_finetuning else None,  # Use None for full fine-tuning to avoid device mapping issues
            }
            if hf_token:
                model_kwargs['token'] = hf_token
                print("Using Hugging Face token for authentication")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(**model_kwargs, model_name=model_name)
            print("Model loaded successfully.")
        except NotImplementedError as e:
            if "GPU" in str(e):
                error_msg = f"GPU is required for unsloth but not available. Error: {e}"
                warnings.warn(error_msg)
                raise NotImplementedError(f"No GPU available for model loading: {error_msg}") from e
            else:
                raise
        except Exception as e:
            print(f"Error loading model: {e}")
            if "401" in str(e):
                print("Hint: This model may require authentication. Please provide a Hugging Face token.")
                print("You can set the HF_TOKEN environment variable or pass hf_token parameter.")
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
            # Check if it's a directory with dataset_info.json (saved using save_to_disk)
            if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
                print("Loading dataset from disk format...")
                from datasets import load_from_disk
                self.dataset = load_from_disk(dataset_path)
            # Load local dataset - check file extension
            elif dataset_path.endswith('.json') or (os.path.isdir(dataset_path) and any(f.endswith('.json') for f in os.listdir(dataset_path))):
                self.dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith('.jsonl') or (os.path.isdir(dataset_path) and any(f.endswith('.jsonl') for f in os.listdir(dataset_path))):
                # Handle JSONL files
                if os.path.isdir(dataset_path):
                    # Find all jsonl files in the directory
                    jsonl_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jsonl')]
                    self.dataset = load_dataset('json', data_files=jsonl_files, split='train')
                else:
                    self.dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith('.parquet') or (os.path.isdir(dataset_path) and any(f.endswith('.parquet') for f in os.listdir(dataset_path))):
                # Handle parquet files
                if os.path.isdir(dataset_path):
                    # Find all parquet files in the directory
                    parquet_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.parquet')]
                    self.dataset = load_dataset('parquet', data_files=parquet_files, split='train')
                else:
                    self.dataset = load_dataset('parquet', data_files=dataset_path, split='train')
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

        # Get the chat template for the tokenizer
        print("Preprocessing data...")
        try:
            # Get the chat template for the tokenizer
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma",  # Use Gemma chat template
            )
            
            def formatting_prompts_func(examples):
                """Format examples into text using chat template."""
                conversations_list = examples.get('conversations', [])
                texts = []
                
                for conversations in conversations_list:
                    if not conversations:
                        texts.append("")
                        continue
                    
                    # Convert to messages format
                    messages = []
                    for conv in conversations:
                        role = conv.get('from', 'user')
                        content = conv.get('value', '')
                        # Map roles to standard format
                        if role == 'human':
                            role = 'user'
                        elif role == 'gpt':
                            role = 'assistant'
                        messages.append({"role": role, "content": content})

                    # Apply chat template
                    try:
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        texts.append(text)
                    except Exception as e:
                        print(f"Error formatting conversation: {e}")
                        texts.append("")
                
                return {"text": texts}

            # Apply formatting to the dataset
            self.dataset = dataset.map(formatting_prompts_func, batched=True)
            print("Data preprocessing completed.")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise

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
        
        # Only setup LoRA if not doing full fine-tuning
        if not self.full_finetuning:
            print("Setting up LoRA adapters...")
            # Setup LoRA for training
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,  # Supports any, but = 0 is optimized
                bias="none",    # Supports any, but = "none" is optimized
                use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None, # And LoftQ
            )
        else:
            print("Using full fine-tuning - all model parameters are trainable")
            # Set environment variable for unsloth
            import os
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"

        # Configure trainer with settings optimized for the fine-tuning mode
        training_args = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": max_steps,
            "learning_rate": 2e-4 if not self.full_finetuning else 1e-5,  # Lower LR for full fine-tuning
            "fp16": False,  # Disable fp16 since model is in bfloat16
            "bf16": True,   # Use bf16 as model is in bfloat16 precision
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": output_dir,
            "report_to": "none",
        }
        
        if self.full_finetuning:
            # Adjust batch size for full fine-tuning (uses more memory)
            training_args["per_device_train_batch_size"] = 1
            training_args["gradient_accumulation_steps"] = 8
            print("Adjusted batch size for full fine-tuning (uses more memory)")
        
        # Configure trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=SFTConfig(**training_args),
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
