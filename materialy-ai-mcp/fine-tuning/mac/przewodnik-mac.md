# Fine-tuning modeli AI na macOS

## Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Przygotowanie środowiska na Mac](#przygotowanie-środowiska-na-mac)
3. [Wykorzystanie Apple Silicon](#wykorzystanie-apple-silicon)
4. [Instalacja narzędzi](#instalacja-narzędzi)
5. [Fine-tuning z Metal Performance Shaders](#fine-tuning-z-metal-performance-shaders)
6. [MLX Framework od Apple](#mlx-framework-od-apple)
7. [Optymalizacja dla macOS](#optymalizacja-dla-macos)
8. [Porównanie Intel vs Apple Silicon](#porównanie-intel-vs-apple-silicon)

## Wprowadzenie

macOS oferuje unikalne możliwości dla fine-tuningu modeli AI, szczególnie na komputerach z chipami Apple Silicon (M1, M2, M3). System wykorzystuje Metal Performance Shaders (MPS) dla akceleracji GPU oraz dedykowany framework MLX optymalizowany pod kątem architektury Apple.

### Wymagania sprzętowe:
- **Apple Silicon**: M1/M2/M3 z min. 16GB RAM (zalecane 32GB+)
- **Intel Mac**: Procesor i7/i9, min. 16GB RAM, AMD GPU (ograniczone wsparcie)
- **macOS**: Ventura 13.0 lub nowszy
- **Dysk**: Min. 100GB wolnego miejsca

## Przygotowanie środowiska na Mac

### 1. Instalacja Homebrew

```bash
# Instalacja Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Dodaj do PATH (dla Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Instalacja Python przez Homebrew

```bash
# Instalacja Python 3.11
brew install python@3.11

# Weryfikacja
python3 --version
pip3 --version

# Utwórz alias
echo 'alias python=python3' >> ~/.zshrc
echo 'alias pip=pip3' >> ~/.zshrc
source ~/.zshrc
```

### 3. Narzędzia developerskie

```bash
# Xcode Command Line Tools
xcode-select --install

# Dodatkowe narzędzia
brew install cmake libomp git-lfs
```

## Wykorzystanie Apple Silicon

### Metal Performance Shaders (MPS)

```python
# check_mps.py
import torch

# Sprawdź dostępność MPS
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) jest dostępny!")
    device = torch.device("mps")
    
    # Informacje o GPU
    print(f"Używany device: {device}")
    
    # Test wydajności
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print("Test mnożenia macierzy zakończony pomyślnie")
else:
    print("MPS niedostępny, używam CPU")
    device = torch.device("cpu")
```

### Unified Memory Architecture

```python
# memory_management.py
import psutil
import subprocess

def get_system_info():
    # RAM
    ram = psutil.virtual_memory()
    print(f"Całkowita pamięć: {ram.total / (1024**3):.2f} GB")
    print(f"Dostępna pamięć: {ram.available / (1024**3):.2f} GB")
    
    # Informacje o chipie (Apple Silicon)
    try:
        chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'])
        print(f"Chip: {chip_info.decode().strip()}")
    except:
        print("Nie mogę określić typu chipa")
```

## Instalacja narzędzi

### Środowisko wirtualne

```bash
# Utwórz środowisko
python -m venv ai_env
source ai_env/bin/activate

# Aktualizuj pip
pip install --upgrade pip setuptools wheel
```

### PyTorch z obsługą MPS

```bash
# PyTorch dla Apple Silicon
pip install torch torchvision torchaudio

# Weryfikacja MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Biblioteki do fine-tuningu

```bash
# Transformers i narzędzia
pip install transformers datasets accelerate peft
pip install bitsandbytes-macos  # Specjalna wersja dla Mac
pip install wandb tensorboard jupyterlab

# Biblioteki pomocnicze
pip install pandas numpy matplotlib seaborn tqdm scikit-learn
```

## Fine-tuning z Metal Performance Shaders

### 1. Konfiguracja dla MPS

```python
# mps_config.py
import torch
from transformers import TrainingArguments

def get_mps_config():
    # Sprawdź dostępność MPS
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS nie jest dostępny na tym urządzeniu")
    
    # Konfiguracja treningowa dla MPS
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Dostosuj do pamięci
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        # MPS nie wspiera fp16, użyj fp32
        fp16=False,
        bf16=False,  # Brak wsparcia dla bf16 na MPS
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        # Użyj MPS
        use_mps_device=True,
        dataloader_num_workers=0,  # MPS wymaga 0 workers
    )
    
    return training_args
```

### 2. Model loading dla MPS

```python
# mps_model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_for_mps(model_name):
    # Specjalne ustawienia dla MPS
    device = torch.device("mps")
    
    # Załaduj model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # MPS wymaga float32
        low_cpu_mem_usage=True,
        device_map={"": device}
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Przenieś model na MPS
    model = model.to(device)
    
    return model, tokenizer, device
```

### 3. Trening z MPS

```python
# train_mps.py
from transformers import Trainer, DataCollatorForLanguageModeling
import torch

class MPSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _move_model_to_device(self, model, device):
        # Specjalna obsługa dla MPS
        if device.type == "mps":
            model = model.to(device)
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Upewnij się, że wszystko jest na MPS
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        return super().compute_loss(model, inputs, return_outputs)

# Użycie
def train_on_mps(model, tokenizer, dataset):
    device = torch.device("mps")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = MPSTrainer(
        model=model,
        args=get_mps_config(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Trening
    trainer.train()
    
    return trainer
```

## MLX Framework od Apple

### 1. Instalacja MLX

```bash
# MLX - framework Apple dla ML
pip install mlx mlx-lm

# Weryfikacja
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
```

### 2. Fine-tuning z MLX

```python
# mlx_finetune.py
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

class MLXFineTuner:
    def __init__(self, model_path):
        self.model, self.tokenizer = load(model_path)
        
    def prepare_data(self, dataset):
        # Konwersja danych do formatu MLX
        def tokenize(text):
            return self.tokenizer.encode(text)
        
        # Tokenizacja
        tokenized = []
        for item in dataset:
            tokens = tokenize(item['text'])
            tokenized.append(mx.array(tokens))
        
        return tokenized
    
    def finetune(self, data, learning_rate=1e-5, epochs=3):
        # Optimizer MLX
        optimizer = mx.optimizers.Adam(learning_rate=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in self.get_batches(data):
                # Forward pass
                loss = self.model.loss(batch)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")
    
    def get_batches(self, data, batch_size=4):
        # Implementacja batch loadera
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]
```

### 3. Konwersja modeli do MLX

```python
# convert_to_mlx.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import convert

def convert_hf_to_mlx(hf_model_name, output_path):
    # Konwersja Hugging Face -> MLX
    convert.convert_hf_model(
        hf_model_name,
        output_path,
        quantize=True,  # 4-bit quantization
    )
    
    print(f"Model skonwertowany do: {output_path}")

# Użycie
convert_hf_to_mlx("microsoft/phi-2", "./phi2-mlx")
```

## Optymalizacja dla macOS

### 1. Wykorzystanie Neural Engine

```python
# neural_engine.py
import coremltools as ct
from transformers import TFAutoModelForCausalLM

def convert_to_coreml(model_name):
    # Załaduj model TensorFlow
    tf_model = TFAutoModelForCausalLM.from_pretrained(model_name)
    
    # Konwersja do Core ML
    mlmodel = ct.convert(
        tf_model,
        source="tensorflow",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13
    )
    
    # Zapisz
    mlmodel.save(f"{model_name.split('/')[-1]}.mlpackage")
```

### 2. Memory Pressure Management

```python
# memory_manager.py
import resource
import gc

class MemoryManager:
    def __init__(self, max_memory_gb=None):
        if max_memory_gb:
            # Ustaw limit pamięci
            max_memory = max_memory_gb * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    
    def optimize_memory(self):
        # Wyczyść cache
        gc.collect()
        
        # Dla MPS
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    
    def monitor_memory(self):
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        print(f"RSS Memory: {memory_info.rss / 1024**3:.2f} GB")
        print(f"VMS Memory: {memory_info.vms / 1024**3:.2f} GB")
```

### 3. Batch Processing Optimization

```python
# batch_optimizer.py
def optimize_batch_size(model, sample_input, max_memory_gb=16):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    optimal_batch = 1
    
    for batch_size in batch_sizes:
        try:
            # Testuj batch size
            dummy_input = {
                k: v.repeat(batch_size, 1) if v.dim() > 1 else v.repeat(batch_size)
                for k, v in sample_input.items()
            }
            
            with torch.no_grad():
                _ = model(**dummy_input)
            
            # Jeśli działa, zapisz
            optimal_batch = batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
    
    print(f"Optymalny batch size: {optimal_batch}")
    return optimal_batch
```

## Porównanie Intel vs Apple Silicon

### Benchmark Performance

```python
# benchmark.py
import time
import torch
import numpy as np

def benchmark_device(device_name, device):
    print(f"\n=== Benchmark dla {device_name} ===")
    
    # Test 1: Mnożenie macierzy
    size = 5000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start = time.time()
    c = torch.matmul(a, b)
    torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
    matmul_time = time.time() - start
    
    print(f"Mnożenie macierzy {size}x{size}: {matmul_time:.4f}s")
    
    # Test 2: Transformer forward pass
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    input_ids = torch.randint(0, 1000, (32, 512), device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    
    # Test
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
    transformer_time = (time.time() - start) / 10
    
    print(f"BERT forward pass (batch=32): {transformer_time:.4f}s")
    
    return {
        "matmul": matmul_time,
        "transformer": transformer_time
    }

# Porównanie
if torch.backends.mps.is_available():
    mps_results = benchmark_device("Apple Silicon (MPS)", torch.device("mps"))
    cpu_results = benchmark_device("CPU", torch.device("cpu"))
    
    print("\n=== Podsumowanie ===")
    print(f"MPS vs CPU speedup (matmul): {cpu_results['matmul']/mps_results['matmul']:.2f}x")
    print(f"MPS vs CPU speedup (transformer): {cpu_results['transformer']/mps_results['transformer']:.2f}x")
```

### Wskazówki dla różnych chipów

```python
# chip_specific_config.py
import subprocess
import platform

def get_optimal_config():
    # Określ typ chipa
    chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
    
    config = {
        "device": "cpu",
        "batch_size": 4,
        "gradient_accumulation": 4,
        "mixed_precision": False
    }
    
    if "Apple M1" in chip:
        config.update({
            "device": "mps",
            "batch_size": 4,
            "gradient_accumulation": 8,
            "notes": "M1: 8-16GB unified memory, good for small models"
        })
    elif "Apple M2" in chip:
        config.update({
            "device": "mps",
            "batch_size": 8,
            "gradient_accumulation": 4,
            "notes": "M2: Better efficiency, handle medium models"
        })
    elif "Apple M3" in chip:
        config.update({
            "device": "mps",
            "batch_size": 16,
            "gradient_accumulation": 2,
            "notes": "M3: Dynamic caching, best for larger models"
        })
    elif "Intel" in chip:
        config.update({
            "device": "cpu",
            "batch_size": 2,
            "gradient_accumulation": 16,
            "notes": "Intel Mac: Limited to CPU, use small batches"
        })
    
    return config
```

## Skrypty pomocnicze dla Mac

### Automatyzacja setup

```bash
#!/bin/bash
# setup_mac_ai.sh

echo "🚀 Konfiguracja środowiska AI na macOS"

# Sprawdź architekturę
ARCH=$(uname -m)
echo "Architektura: $ARCH"

# Instalacja zależności
if ! command -v brew &> /dev/null; then
    echo "Instaluję Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Python i narzędzia
brew install python@3.11 cmake libomp

# Środowisko wirtualne
python3.11 -m venv ~/ai_env
source ~/ai_env/bin/activate

# Instalacja pakietów
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate mlx mlx-lm

# Test MPS
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS dostępny!')
else:
    print('❌ MPS niedostępny')
"

echo "✅ Konfiguracja zakończona!"
```

### Monitor treningu dla Mac

```python
# mac_training_monitor.py
import subprocess
import psutil
import time
from datetime import datetime

class MacTrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        
    def get_gpu_stats(self):
        # Użyj powermetrics dla Apple Silicon
        try:
            cmd = ['sudo', 'powermetrics', '--samplers', 'gpu_power', '-i', '1', '-n', '1']
            output = subprocess.check_output(cmd).decode()
            # Parse output for GPU metrics
            return output
        except:
            return "GPU stats unavailable"
    
    def get_thermal_stats(self):
        # Temperatura
        try:
            cmd = ['sudo', 'powermetrics', '--samplers', 'thermal', '-i', '1', '-n', '1']
            output = subprocess.check_output(cmd).decode()
            return output
        except:
            return "Thermal stats unavailable"
    
    def monitor_training(self, interval=10):
        while True:
            print(f"\n{'='*50}")
            print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Uptime: {(time.time() - self.start_time)/60:.1f} minutes")
            
            # System stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"CPU Usage: {cpu_percent}%")
            print(f"Memory: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
            
            # GPU stats (requires sudo)
            # print(self.get_gpu_stats())
            
            time.sleep(interval)
```

## Troubleshooting macOS

### Problem: MPS Memory Errors

```python
# Rozwiązanie 1: Wyczyść cache
torch.mps.empty_cache()
torch.mps.synchronize()

# Rozwiązanie 2: Zmniejsz precyzję
model = model.half()  # float16 zamiast float32

# Rozwiązanie 3: Gradient checkpointing
model.gradient_checkpointing_enable()
```

### Problem: Kernel Crash

```bash
# Reset MPS
sudo killall -9 metalperformanceshaders

# Wyczyść cache systemowy
sudo purge

# Restart Terminal/IDE
```

### Problem: Wolne ładowanie modelu

```python
# Użyj low_cpu_mem_usage
model = AutoModel.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    device_map={"": torch.device("mps")}
)
```

## Podsumowanie

Fine-tuning na macOS, szczególnie z Apple Silicon, oferuje:

1. **Unified Memory** - brak transferu między CPU a GPU
2. **Energy Efficiency** - długie sesje treningowe na baterii
3. **Metal/MPS** - natywna akceleracja GPU
4. **MLX** - zoptymalizowany framework Apple

Kluczowe różnice vs Windows/Linux:
- Brak CUDA, ale MPS oferuje podobną funkcjonalność
- Lepsze zarządzanie pamięcią dla modeli średniej wielkości
- Niższa wydajność dla bardzo dużych modeli vs high-end GPU NVIDIA

Przejdź do [przykładów kodu](../../przyklady/README.md) lub wróć do [wprowadzenia](../../wprowadzenie/podstawy-ai-mcp.md).