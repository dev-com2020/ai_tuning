#!/usr/bin/env python3
"""
Przykłady fine-tuningu modeli AI
Kompatybilne z Windows, macOS i Linux
"""

import torch
import platform
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import json
from typing import Dict, List, Optional, Union


class UniversalFineTuner:
    """Uniwersalny fine-tuner działający na różnych platformach"""
    
    def __init__(self, model_name: str, output_dir: str = "./fine_tuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        
    def _get_device(self) -> torch.device:
        """Automatyczne wykrywanie najlepszego urządzenia"""
        if torch.cuda.is_available():
            print("🎮 Używam GPU NVIDIA (CUDA)")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("🍎 Używam Apple Silicon (MPS)")
            return torch.device("mps")
        else:
            print("💻 Używam CPU")
            return torch.device("cpu")
    
    def load_model(self, use_lora: bool = True, lora_r: int = 16):
        """Załaduj model z opcjonalnym LoRA"""
        print(f"📥 Ładowanie modelu: {self.model_name}")
        
        # Konfiguracja dla różnych platform
        device_config = self._get_device_config()
        
        # Załaduj tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Załaduj model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **device_config
        )
        
        # Zastosuj LoRA jeśli włączone
        if use_lora:
            print("🔧 Konfigurowanie LoRA...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def _get_device_config(self) -> Dict:
        """Konfiguracja specyficzna dla urządzenia"""
        if self.device.type == "cuda":
            return {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": True
            }
        elif self.device.type == "mps":
            return {
                "torch_dtype": torch.float32,  # MPS wymaga float32
                "low_cpu_mem_usage": True,
                "device_map": {"": self.device}
            }
        else:
            return {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True
            }
    
    def prepare_dataset(self, data_path: str, format_type: str = "instruction"):
        """Przygotuj dataset do treningu"""
        print(f"📊 Przygotowywanie datasetu z: {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path).to_dict('records')
        else:
            raise ValueError("Wspierane formaty: .json, .csv")
        
        # Formatowanie w zależności od typu
        if format_type == "instruction":
            formatted_data = [self._format_instruction(item) for item in data]
        elif format_type == "chat":
            formatted_data = [self._format_chat(item) for item in data]
        else:
            formatted_data = data
        
        # Konwersja do Dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenizacja
        tokenized_dataset = dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            ),
            batched=True
        )
        
        return tokenized_dataset
    
    def _format_instruction(self, item: Dict) -> Dict:
        """Format dla treningu instrukcji"""
        text = f"""### Instrukcja:
{item.get('instruction', '')}

### Wejście:
{item.get('input', '')}

### Odpowiedź:
{item.get('output', '')}"""
        return {"text": text}
    
    def _format_chat(self, item: Dict) -> Dict:
        """Format dla treningu konwersacyjnego"""
        messages = item.get('messages', [])
        text = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            text += f"<|{role}|>\n{content}\n"
        return {"text": text}
    
    def train(self, train_dataset, eval_dataset=None, epochs: int = 3):
        """Przeprowadź trening"""
        print("🚀 Rozpoczynam trening...")
        
        # Argumenty treningowe dostosowane do platformy
        training_args = self._get_training_args(epochs)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Trening
        trainer.train()
        
        # Zapisz model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✅ Model zapisany w: {self.output_dir}")
        
        return trainer
    
    def _get_training_args(self, epochs: int) -> TrainingArguments:
        """Argumenty treningowe dostosowane do platformy"""
        base_args = {
            "output_dir": self.output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "save_steps": 100,
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "report_to": "tensorboard",
        }
        
        # Dostosowania platform-specific
        if self.device.type == "cuda":
            base_args.update({
                "fp16": True,
                "dataloader_num_workers": 4,
            })
        elif self.device.type == "mps":
            base_args.update({
                "use_mps_device": True,
                "dataloader_num_workers": 0,  # MPS wymaga 0
                "fp16": False,  # MPS nie wspiera fp16
            })
        else:
            base_args.update({
                "dataloader_num_workers": 2,
            })
        
        return TrainingArguments(**base_args)
    
    def inference(self, prompt: str, max_length: int = 100):
        """Generuj tekst używając wytrenowanego modelu"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Przykład 1: Fine-tuning małego modelu na własnych danych
def example_small_model_finetuning():
    """Przykład fine-tuningu Microsoft Phi-2"""
    print("\n" + "="*50)
    print("Przykład 1: Fine-tuning małego modelu")
    print("="*50)
    
    # Przygotuj przykładowe dane
    sample_data = [
        {
            "instruction": "Wyjaśnij czym jest Python",
            "input": "",
            "output": "Python to wysokopoziomowy język programowania..."
        },
        {
            "instruction": "Napisz funkcję sortującą listę",
            "input": "lista = [3, 1, 4, 1, 5, 9]",
            "output": "def sortuj_liste(lista):\n    return sorted(lista)"
        }
    ]
    
    # Zapisz dane
    with open("sample_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False)
    
    # Inicjalizuj fine-tuner
    tuner = UniversalFineTuner("microsoft/phi-2", output_dir="./phi2_finetuned")
    
    # Załaduj model
    tuner.load_model(use_lora=True)
    
    # Przygotuj dataset
    dataset = tuner.prepare_dataset("sample_data.json", format_type="instruction")
    
    # Podziel na train/eval
    split_dataset = dataset.train_test_split(test_size=0.2)
    
    # Trenuj
    tuner.train(
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        epochs=1  # Mało epok dla przykładu
    )
    
    # Test
    response = tuner.inference("### Instrukcja:\nCo to jest rekurencja?\n\n### Odpowiedź:\n")
    print(f"\nOdpowiedź modelu: {response}")


# Przykład 2: Fine-tuning dla zadania klasyfikacji
def example_classification_finetuning():
    """Przykład fine-tuningu do klasyfikacji sentymentu"""
    print("\n" + "="*50)
    print("Przykład 2: Fine-tuning dla klasyfikacji")
    print("="*50)
    
    from transformers import AutoModelForSequenceClassification
    
    # Dane treningowe
    sentiment_data = [
        {"text": "Ten produkt jest świetny!", "label": 1},
        {"text": "Bardzo słaba jakość, nie polecam", "label": 0},
        {"text": "Najlepszy zakup w tym roku", "label": 1},
        {"text": "Totalna porażka, pieniądze wyrzucone w błoto", "label": 0},
    ]
    
    # Model do klasyfikacji
    model_name = "allegro/herbert-base-cased"  # Polski BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Tokenizacja
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # Dataset
    dataset = Dataset.from_list(sentiment_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print("✅ Model klasyfikacji przygotowany")


# Przykład 3: Quantization dla efektywności
def example_quantization():
    """Przykład kwantyzacji modelu"""
    print("\n" + "="*50)
    print("Przykład 3: Kwantyzacja modelu")
    print("="*50)
    
    from transformers import BitsAndBytesConfig
    
    # Konfiguracja 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Załaduj model z kwantyzacją
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Sprawdź rozmiar
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Liczba parametrów: {param_size:,}")
    print(f"Rozmiar w pamięci: ~{param_size * 4 / 1024**3:.2f} GB (4-bit)")


# Przykład 4: Multi-GPU training
def example_multi_gpu():
    """Przykład treningu na wielu GPU"""
    print("\n" + "="*50)
    print("Przykład 4: Multi-GPU Training")
    print("="*50)
    
    if torch.cuda.device_count() > 1:
        print(f"Wykryto {torch.cuda.device_count()} GPU")
        
        # Konfiguracja dla multi-GPU
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        
        # Model będzie automatycznie dystrybuowany
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = accelerator.prepare(model)
        
        print("✅ Model przygotowany do treningu multi-GPU")
    else:
        print("❌ Tylko jedno GPU dostępne")


# Przykład 5: Monitoring treningu
class TrainingMonitor:
    """Monitor postępu treningu"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.metrics = []
        
    def log_metrics(self, epoch: int, loss: float, learning_rate: float):
        """Loguj metryki"""
        metric = {
            "epoch": epoch,
            "loss": loss,
            "learning_rate": learning_rate,
            "timestamp": pd.Timestamp.now()
        }
        self.metrics.append(metric)
        
        # Zapisz do CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(f"{self.log_dir}/training_metrics.csv", index=False)
        
    def plot_loss(self):
        """Wykres straty"""
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.metrics)
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(f"{self.log_dir}/loss_plot.png")
        plt.close()


# Główna funkcja
def main():
    parser = argparse.ArgumentParser(description="Przykłady fine-tuningu")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Który przykład uruchomić"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_small_model_finetuning,
        2: example_classification_finetuning,
        3: example_quantization,
        4: example_multi_gpu,
        5: lambda: print("Użyj klasy TrainingMonitor w swoim kodzie")
    }
    
    # Informacje o systemie
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # Uruchom przykład
    examples[args.example]()


if __name__ == "__main__":
    main()