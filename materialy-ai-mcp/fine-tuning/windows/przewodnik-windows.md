# Fine-tuning modeli AI na Windows

## Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Przygotowanie środowiska](#przygotowanie-środowiska)
3. [Instalacja narzędzi](#instalacja-narzędzi)
4. [Fine-tuning z Hugging Face](#fine-tuning-z-hugging-face)
5. [Fine-tuning OpenAI GPT](#fine-tuning-openai-gpt)
6. [Fine-tuning lokalnych modeli](#fine-tuning-lokalnych-modeli)
7. [Optymalizacja wydajności](#optymalizacja-wydajności)
8. [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## Wprowadzenie

Fine-tuning to proces dostosowywania wstępnie wytrenowanego modelu AI do konkretnych zadań lub domen. Na Windows możemy przeprowadzić fine-tuning zarówno modeli chmurowych (OpenAI, Anthropic), jak i lokalnych (LLaMA, Mistral).

### Wymagania sprzętowe:
- **CPU**: Intel i5/AMD Ryzen 5 lub lepszy
- **RAM**: Minimum 16GB (zalecane 32GB+)
- **GPU**: NVIDIA z min. 8GB VRAM (dla lokalnego treningu)
- **Dysk**: Min. 100GB wolnego miejsca

## Przygotowanie środowiska

### 1. Instalacja Python

```powershell
# Pobierz Python 3.10 lub 3.11 z python.org
# Podczas instalacji zaznacz "Add Python to PATH"

# Sprawdź instalację
python --version
pip --version
```

### 2. Instalacja CUDA (dla GPU NVIDIA)

1. Sprawdź kompatybilność GPU: https://developer.nvidia.com/cuda-gpus
2. Pobierz CUDA Toolkit 11.8 lub 12.1
3. Zainstaluj cuDNN odpowiedni dla wersji CUDA
4. Zweryfikuj instalację:

```powershell
nvidia-smi
nvcc --version
```

### 3. Konfiguracja środowiska wirtualnego

```powershell
# Utwórz środowisko wirtualne
python -m venv ai_finetuning
.\ai_finetuning\Scripts\activate

# Aktualizuj pip
python -m pip install --upgrade pip
```

## Instalacja narzędzi

### Podstawowe biblioteki

```powershell
# Instalacja PyTorch z obsługą CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Biblioteki do fine-tuningu
pip install transformers datasets accelerate peft bitsandbytes
pip install wandb tensorboard jupyter notebook

# Narzędzia pomocnicze
pip install pandas numpy matplotlib seaborn tqdm
```

### Weryfikacja instalacji

```python
# test_setup.py
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Fine-tuning z Hugging Face

### 1. Przygotowanie danych

```python
# prepare_data.py
from datasets import Dataset, DatasetDict
import pandas as pd

# Załaduj dane treningowe
def prepare_dataset(train_file, val_file):
    # Wczytaj dane
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    # Konwertuj do formatu Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

# Format danych dla instrukcji
def format_instruction(example):
    return {
        'text': f"""### Instrukcja:
{example['instruction']}

### Wejście:
{example['input']}

### Odpowiedź:
{example['output']}"""
    }
```

### 2. Konfiguracja treningu

```python
# config.py
from transformers import TrainingArguments
from peft import LoraConfig

# Parametry LoRA dla efektywnego fine-tuningu
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Argumenty treningowe
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,  # Użyj fp16 dla oszczędności pamięci
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to="tensorboard",
)
```

### 3. Proces treningu

```python
# train.py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training
import torch

def train_model(model_name, dataset, lora_config, training_args):
    # Załaduj model i tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,  # Kwantyzacja 4-bit
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Przygotuj model do treningu
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Tokenizuj dane
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Utwórz trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )
    
    # Rozpocznij trening
    trainer.train()
    
    # Zapisz model
    trainer.save_model("./finetuned_model")
    return model, tokenizer
```

## Fine-tuning OpenAI GPT

### 1. Przygotowanie danych w formacie JSONL

```python
# prepare_openai_data.py
import json

def prepare_openai_dataset(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Format dla GPT-3.5/4
            formatted = {
                "messages": [
                    {"role": "system", "content": "Jesteś pomocnym asystentem."},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]}
                ]
            }
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
```

### 2. Upload i fine-tuning przez API

```python
# openai_finetune.py
from openai import OpenAI
import time

client = OpenAI(api_key="your-api-key")

# Upload pliku treningowego
def upload_training_file(file_path):
    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    return response.id

# Rozpocznij fine-tuning
def start_finetuning(training_file_id, model="gpt-3.5-turbo"):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 4,
            "learning_rate_multiplier": 2
        }
    )
    return response.id

# Monitoruj postęp
def monitor_job(job_id):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {job.status}")
        
        if job.status in ["succeeded", "failed", "cancelled"]:
            break
            
        time.sleep(60)
    
    return job
```

## Fine-tuning lokalnych modeli

### 1. LLaMA 2/3 Fine-tuning

```python
# llama_finetune.py
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

def finetune_llama(model_path, dataset):
    # Specjalne ustawienia dla LLaMA
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    # Dodaj specjalne tokeny jeśli potrzeba
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    })
    model.resize_token_embeddings(len(tokenizer))
    
    # Reszta procesu jak wyżej...
```

### 2. Mistral Fine-tuning

```python
# mistral_finetune.py
def finetune_mistral():
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # Mistral wymaga specjalnej uwagi na sliding window attention
    model_config = {
        "sliding_window": 4096,
        "max_position_embeddings": 32768,
    }
    
    # Użyj Flash Attention 2 dla lepszej wydajności
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        **model_config
    )
```

## Optymalizacja wydajności

### 1. Gradient Checkpointing

```python
# Oszczędność pamięci GPU
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# W pętli treningowej
with autocast():
    outputs = model(**inputs)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. DeepSpeed Integration

```powershell
# Instalacja
pip install deepspeed

# Konfiguracja w ds_config.json
```

```json
{
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "gradient_accumulation_steps": 4,
    "train_batch_size": 16,
    "gradient_clipping": 1.0
}
```

### 4. Monitorowanie zasobów

```python
# monitor.py
import psutil
import GPUtil

def monitor_resources():
    # CPU i RAM
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    
    # GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        print(f"  Utilization: {gpu.load*100}%")
        print(f"  Temperature: {gpu.temperature}°C")
```

## Rozwiązywanie problemów

### Problem: Out of Memory (OOM)

```python
# Rozwiązania:
# 1. Zmniejsz batch size
training_args.per_device_train_batch_size = 1

# 2. Użyj gradient accumulation
training_args.gradient_accumulation_steps = 16

# 3. Włącz gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Użyj kwantyzacji
load_in_8bit=True  # lub load_in_4bit=True
```

### Problem: Wolny trening

```python
# 1. Sprawdź wykorzystanie GPU
torch.cuda.synchronize()
start = time.time()
# ... operacja ...
torch.cuda.synchronize()
print(f"Czas: {time.time() - start}s")

# 2. Użyj DataLoader z większą liczbą workerów
DataLoader(dataset, num_workers=4, pin_memory=True)

# 3. Kompiluj model (PyTorch 2.0+)
model = torch.compile(model)
```

### Problem: Błędy CUDA

```powershell
# Wyczyść cache CUDA
python -c "import torch; torch.cuda.empty_cache()"

# Sprawdź wersje
python -c "import torch; print(torch.version.cuda)"

# Reinstalacja z odpowiednią wersją CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Skrypty pomocnicze

### Automatyczny fine-tuning

```powershell
# run_finetuning.ps1
param(
    [string]$model = "microsoft/phi-2",
    [string]$data = "./data/train.csv",
    [int]$epochs = 3
)

# Aktywuj środowisko
& .\ai_finetuning\Scripts\activate

# Sprawdź GPU
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Uruchom trening
python train.py `
    --model_name $model `
    --train_data $data `
    --num_epochs $epochs `
    --output_dir "./results" `
    --logging_dir "./logs"

# Ewaluacja
python evaluate.py --model_path "./results/best_model"
```

### Monitorowanie treningu

```python
# monitor_training.py
from tensorboard import notebook
import subprocess

# Uruchom TensorBoard
log_dir = "./logs"
subprocess.Popen(["tensorboard", "--logdir", log_dir])

# Lub w Jupyter Notebook
# %load_ext tensorboard
# %tensorboard --logdir logs
```

## Podsumowanie

Fine-tuning na Windows wymaga odpowiedniego przygotowania środowiska, ale oferuje pełną kontrolę nad procesem. Kluczowe jest:

1. **Wybór odpowiedniego sprzętu** - GPU NVIDIA znacząco przyspiesza trening
2. **Optymalizacja pamięci** - użycie technik jak LoRA, kwantyzacja
3. **Monitorowanie procesu** - TensorBoard, Weights & Biases
4. **Iteracyjne podejście** - zacznij od małych modeli i datasetów

Przejdź do [przewodnika dla Mac](../mac/przewodnik-mac.md) lub zobacz [przykłady kodu](../../przyklady/fine-tuning-examples.py).