# Narzędzia pomocnicze AI/MCP

Ten katalog zawiera narzędzia ułatwiające pracę z AI i MCP na różnych systemach operacyjnych.

## 📋 Zawartość

### 1. setup-environment.py
Uniwersalny skrypt do konfiguracji środowiska AI/MCP.

**Funkcje:**
- Automatyczne wykrywanie systemu (Windows/Mac/Linux)
- Sprawdzanie dostępności GPU (NVIDIA CUDA, Apple MPS)
- Instalacja pakietów Python
- Konfiguracja Node.js dla MCP
- Tworzenie środowisk wirtualnych
- Generowanie skryptów testowych

**Użycie:**
```bash
# Pełna instalacja
python setup-environment.py --full

# Tylko pakiety Python
python setup-environment.py --python-only

# Tylko serwer MCP
python setup-environment.py --mcp-only

# Z testami
python setup-environment.py --full --test
```

## 🛠️ Dodatkowe skrypty

### Benchmark GPU (benchmark_gpu.py)
```python
#!/usr/bin/env python3
"""Benchmark wydajności GPU dla różnych platform"""

import torch
import time
import numpy as np
from typing import Dict, Any

def benchmark_matmul(device: torch.device, size: int = 5000) -> float:
    """Test mnożenia macierzy"""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    return (time.time() - start) / 10

def benchmark_transformer(device: torch.device, batch_size: int = 32) -> float:
    """Test modelu Transformer"""
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (batch_size, 512), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    return (time.time() - start) / 10

def run_benchmarks() -> Dict[str, Any]:
    """Uruchom wszystkie benchmarki"""
    results = {}
    
    # Wykryj dostępne urządzenia
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n🔍 Benchmarking {device_name.upper()}...")
        
        results[device_name] = {
            "matmul_5000": benchmark_matmul(device, 5000),
            "transformer_batch32": benchmark_transformer(device, 32)
        }
        
        print(f"  Matrix multiplication (5000x5000): {results[device_name]['matmul_5000']:.4f}s")
        print(f"  BERT forward pass (batch=32): {results[device_name]['transformer_batch32']:.4f}s")
    
    # Porównanie
    if len(devices) > 1:
        print("\n📊 Porównanie wydajności:")
        base_device = "cpu"
        for device in devices:
            if device != base_device:
                matmul_speedup = results[base_device]["matmul_5000"] / results[device]["matmul_5000"]
                transformer_speedup = results[base_device]["transformer_batch32"] / results[device]["transformer_batch32"]
                print(f"  {device.upper()} vs CPU:")
                print(f"    Matrix multiplication: {matmul_speedup:.2f}x faster")
                print(f"    Transformer: {transformer_speedup:.2f}x faster")
    
    return results

if __name__ == "__main__":
    print("🚀 GPU Benchmark Tool")
    print("=" * 50)
    results = run_benchmarks()
    
    # Zapisz wyniki
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n💾 Wyniki zapisane w benchmark_results.json")
```

### Monitor zasobów (resource_monitor.py)
```python
#!/usr/bin/env python3
"""Monitor zasobów podczas treningu modeli"""

import psutil
import GPUtil
import time
import json
from datetime import datetime
from typing import Dict, List
import threading

class ResourceMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.data: List[Dict] = []
        
    def get_current_stats(self) -> Dict:
        """Pobierz aktualne statystyki"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count()
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / (1024**3),
                "used_gb": psutil.disk_usage('/').used / (1024**3),
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        # GPU stats
        try:
            gpus = GPUtil.getGPUs()
            stats["gpu"] = []
            for gpu in gpus:
                stats["gpu"].append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_percent": gpu.memoryUtil * 100,
                    "gpu_util_percent": gpu.load * 100,
                    "temperature": gpu.temperature
                })
        except:
            stats["gpu"] = []
        
        return stats
    
    def start_monitoring(self):
        """Rozpocznij monitorowanie"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        print("📊 Monitoring started...")
    
    def stop_monitoring(self):
        """Zatrzymaj monitorowanie"""
        self.monitoring = False
        self.monitor_thread.join()
        print("📊 Monitoring stopped.")
    
    def _monitor_loop(self):
        """Główna pętla monitorowania"""
        while self.monitoring:
            stats = self.get_current_stats()
            self.data.append(stats)
            time.sleep(self.interval)
    
    def save_report(self, filename: str = "resource_usage.json"):
        """Zapisz raport"""
        with open(filename, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"💾 Report saved to {filename}")
    
    def print_summary(self):
        """Wyświetl podsumowanie"""
        if not self.data:
            print("No data collected")
            return
        
        print("\n📈 Resource Usage Summary:")
        print("=" * 50)
        
        # CPU
        cpu_usage = [d["cpu"]["percent"] for d in self.data]
        print(f"CPU Usage: avg={np.mean(cpu_usage):.1f}%, max={np.max(cpu_usage):.1f}%")
        
        # Memory
        mem_usage = [d["memory"]["percent"] for d in self.data]
        print(f"Memory Usage: avg={np.mean(mem_usage):.1f}%, max={np.max(mem_usage):.1f}%")
        
        # GPU
        if self.data[0]["gpu"]:
            for gpu_id in range(len(self.data[0]["gpu"])):
                gpu_util = [d["gpu"][gpu_id]["gpu_util_percent"] for d in self.data if len(d["gpu"]) > gpu_id]
                gpu_mem = [d["gpu"][gpu_id]["memory_percent"] for d in self.data if len(d["gpu"]) > gpu_id]
                print(f"GPU {gpu_id} Usage: avg={np.mean(gpu_util):.1f}%, max={np.max(gpu_util):.1f}%")
                print(f"GPU {gpu_id} Memory: avg={np.mean(gpu_mem):.1f}%, max={np.max(gpu_mem):.1f}%")

# Przykład użycia
if __name__ == "__main__":
    import numpy as np
    
    monitor = ResourceMonitor(interval=0.5)
    
    # Symulacja obciążenia
    print("🔥 Simulating workload for 10 seconds...")
    monitor.start_monitoring()
    
    # Tu normalnie byłby kod treningu
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Symulacja obciążenia
    for i in range(10):
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        time.sleep(1)
        print(f"  Step {i+1}/10")
    
    monitor.stop_monitoring()
    monitor.print_summary()
    monitor.save_report()
```

### Konwerter modeli (model_converter.py)
```python
#!/usr/bin/env python3
"""Konwerter modeli między różnymi formatami"""

import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import onnx
import coremltools as ct
from typing import Optional, Dict, Any

class ModelConverter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Załaduj model i tokenizer"""
        print(f"📥 Loading model from {self.model_path}")
        self.model = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print("✅ Model loaded successfully")
        
    def to_onnx(self, output_path: str, opset_version: int = 14):
        """Konwertuj do ONNX"""
        print("🔄 Converting to ONNX...")
        
        # Przykładowe wejście
        dummy_input = self.tokenizer(
            "Example text for conversion",
            return_tensors="pt"
        )
        
        # Export
        torch.onnx.export(
            self.model,
            tuple(dummy_input.values()),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=opset_version
        )
        
        print(f"✅ ONNX model saved to {output_path}")
        
    def to_coreml(self, output_path: str):
        """Konwertuj do Core ML (dla macOS/iOS)"""
        print("🔄 Converting to Core ML...")
        
        # Najpierw konwertuj do ONNX
        onnx_path = "temp_model.onnx"
        self.to_onnx(onnx_path)
        
        # Konwertuj ONNX do Core ML
        mlmodel = ct.convert(
            onnx_path,
            minimum_deployment_target=ct.target.macOS13
        )
        
        mlmodel.save(output_path)
        print(f"✅ Core ML model saved to {output_path}")
        
        # Usuń tymczasowy plik
        Path(onnx_path).unlink()
        
    def to_tflite(self, output_path: str):
        """Konwertuj do TensorFlow Lite"""
        print("🔄 Converting to TFLite...")
        
        # Ta funkcja wymaga TensorFlow
        try:
            import tensorflow as tf
            
            # Konwersja przez ONNX i TF
            # ... implementacja ...
            
            print(f"✅ TFLite model saved to {output_path}")
        except ImportError:
            print("❌ TensorFlow not installed. Install with: pip install tensorflow")
            
    def optimize_for_mobile(self, output_path: str):
        """Optymalizuj model dla urządzeń mobilnych"""
        print("📱 Optimizing for mobile...")
        
        # Kwantyzacja
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Zapisz
        torch.save(quantized_model.state_dict(), output_path)
        print(f"✅ Optimized model saved to {output_path}")

# Przykład użycia
if __name__ == "__main__":
    converter = ModelConverter("bert-base-uncased")
    converter.load_model()
    
    # Konwertuj do różnych formatów
    converter.to_onnx("model.onnx")
    
    # Dla macOS
    if platform.system() == "Darwin":
        converter.to_coreml("model.mlpackage")
```

## 🚀 Quick Start

1. **Sprawdź swój system:**
   ```bash
   python setup-environment.py
   ```

2. **Zainstaluj wszystko:**
   ```bash
   python setup-environment.py --full
   ```

3. **Uruchom benchmarki:**
   ```bash
   python benchmark_gpu.py
   ```

4. **Monitoruj zasoby podczas treningu:**
   ```python
   from resource_monitor import ResourceMonitor
   
   monitor = ResourceMonitor()
   monitor.start_monitoring()
   
   # Twój kod treningu
   train_model()
   
   monitor.stop_monitoring()
   monitor.save_report()
   ```

## 📝 Wskazówki

### Windows
- Upewnij się, że masz zainstalowane Visual Studio Build Tools
- Dla GPU NVIDIA zainstaluj odpowiednią wersję CUDA
- Użyj Anaconda dla łatwiejszego zarządzania pakietami

### macOS
- Na Apple Silicon używaj MPS zamiast CUDA
- Homebrew ułatwia instalację narzędzi
- MLX Framework jest zoptymalizowany dla Apple Silicon

### Linux
- Większość narzędzi działa natywnie
- Dla GPU NVIDIA dodaj repozytoria CUDA
- Użyj Docker dla izolacji środowiska

## 🔗 Przydatne linki

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [MCP Documentation](https://modelcontextprotocol.io)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)