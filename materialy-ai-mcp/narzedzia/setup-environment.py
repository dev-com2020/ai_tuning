#!/usr/bin/env python3
"""
Uniwersalny skrypt konfiguracji środowiska AI/MCP
Działa na Windows, macOS i Linux
"""

import os
import sys
import platform
import subprocess
import json
import argparse
from pathlib import Path
import shutil
import urllib.request
from typing import Dict, List, Tuple, Optional

class EnvironmentSetup:
    """Klasa do konfiguracji środowiska AI/MCP"""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.python_version = sys.version_info
        self.home = Path.home()
        self.config_dir = self.home / '.ai_mcp_config'
        self.config_dir.mkdir(exist_ok=True)
        
    def check_system(self) -> Dict:
        """Sprawdź informacje o systemie"""
        info = {
            "system": self.system,
            "machine": self.machine,
            "python": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            "cpu_count": os.cpu_count(),
        }
        
        # Sprawdź pamięć RAM
        if self.system == "Darwin":  # macOS
            cmd = ["sysctl", "-n", "hw.memsize"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                ram_bytes = int(result.stdout.strip())
                info["ram_gb"] = ram_bytes / (1024**3)
        elif self.system == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            
            memoryStatus = MEMORYSTATUSEX()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
            info["ram_gb"] = memoryStatus.ullTotalPhys / (1024**3)
        else:  # Linux
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        ram_kb = int(line.split()[1])
                        info["ram_gb"] = ram_kb / (1024**2)
                        break
        
        # Sprawdź GPU
        info["gpu"] = self._check_gpu()
        
        return info
    
    def _check_gpu(self) -> Dict:
        """Sprawdź dostępne GPU"""
        gpu_info = {"available": False, "type": "none"}
        
        # NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_info["available"] = True
                gpu_info["type"] = "nvidia"
                gpu_info["details"] = result.stdout.strip()
        except FileNotFoundError:
            pass
        
        # Apple Silicon GPU (MPS)
        if self.system == "Darwin" and self.machine == "arm64":
            gpu_info["available"] = True
            gpu_info["type"] = "apple_silicon"
            
            # Sprawdź model chipa
            try:
                chip = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                ).stdout.strip()
                gpu_info["details"] = chip
            except:
                gpu_info["details"] = "Apple Silicon"
        
        # AMD GPU (podstawowe sprawdzenie)
        if self.system == "Linux":
            try:
                result = subprocess.run(
                    ["lspci", "|", "grep", "-i", "amd.*vga"],
                    shell=True, capture_output=True, text=True
                )
                if result.stdout:
                    gpu_info["available"] = True
                    gpu_info["type"] = "amd"
                    gpu_info["details"] = "AMD GPU detected"
            except:
                pass
        
        return gpu_info
    
    def install_python_packages(self, packages: List[str], upgrade: bool = False):
        """Instaluj pakiety Pythona"""
        print("📦 Instalowanie pakietów Python...")
        
        # Aktualizuj pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Instaluj pakiety
        for package in packages:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package)
            
            print(f"  Installing {package}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ❌ Błąd instalacji {package}: {result.stderr}")
            else:
                print(f"  ✅ {package} zainstalowany")
    
    def setup_nodejs(self):
        """Instaluj Node.js i npm"""
        print("🟢 Konfiguracja Node.js...")
        
        # Sprawdź czy Node.js jest zainstalowany
        try:
            node_version = subprocess.run(
                ["node", "--version"],
                capture_output=True, text=True
            ).stdout.strip()
            print(f"  ✅ Node.js już zainstalowany: {node_version}")
            return
        except FileNotFoundError:
            pass
        
        if self.system == "Darwin":
            # macOS - użyj Homebrew
            print("  Instaluję Node.js przez Homebrew...")
            subprocess.run(["brew", "install", "node"])
        elif self.system == "Windows":
            # Windows - pobierz installer
            print("  Pobierz Node.js z: https://nodejs.org/")
            print("  Lub użyj winget: winget install OpenJS.NodeJS")
        else:
            # Linux
            print("  Instaluję Node.js...")
            subprocess.run(["sudo", "apt-get", "update"])
            subprocess.run(["sudo", "apt-get", "install", "-y", "nodejs", "npm"])
    
    def create_virtual_environment(self, name: str = "ai_env"):
        """Utwórz środowisko wirtualne"""
        env_path = self.home / name
        
        if env_path.exists():
            print(f"⚠️  Środowisko {name} już istnieje")
            return env_path
        
        print(f"🐍 Tworzenie środowiska wirtualnego: {name}")
        subprocess.run([sys.executable, "-m", "venv", str(env_path)])
        
        # Utwórz skrypt aktywacji
        if self.system == "Windows":
            activate_cmd = f"{env_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {env_path}/bin/activate"
        
        print(f"  ✅ Środowisko utworzone")
        print(f"  Aktywacja: {activate_cmd}")
        
        return env_path
    
    def setup_mcp_server(self, server_path: Path):
        """Konfiguruj serwer MCP"""
        print("🔧 Konfiguracja serwera MCP...")
        
        # Utwórz katalog dla serwera
        server_path.mkdir(parents=True, exist_ok=True)
        
        # Package.json
        package_json = {
            "name": "my-mcp-server",
            "version": "1.0.0",
            "type": "module",
            "dependencies": {
                "@modelcontextprotocol/sdk": "^0.5.0",
                "zod": "^3.22.4"
            },
            "scripts": {
                "start": "node index.js"
            }
        }
        
        with open(server_path / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
        
        # Prosty serwer MCP
        server_code = """import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'my-mcp-server',
  version: '1.0.0',
});

// Dodaj narzędzie
server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'hello',
    description: 'Say hello',
    inputSchema: {
      type: 'object',
      properties: {
        name: { type: 'string' }
      }
    }
  }]
}));

server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'hello') {
    return {
      content: [{
        type: 'text',
        text: `Hello, ${request.params.arguments.name}!`
      }]
    };
  }
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
"""
        
        with open(server_path / "index.js", "w") as f:
            f.write(server_code)
        
        print(f"  ✅ Serwer MCP utworzony w: {server_path}")
        print(f"  Instalacja: cd {server_path} && npm install")
        print(f"  Uruchomienie: npm start")
    
    def configure_cuda(self):
        """Konfiguruj CUDA dla NVIDIA GPU"""
        if self.system != "Windows":
            print("ℹ️  Konfiguracja CUDA tylko dla Windows")
            return
        
        print("🎮 Konfiguracja CUDA...")
        
        # Sprawdź wersję CUDA
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✅ CUDA już zainstalowana")
                print(result.stdout)
                return
        except FileNotFoundError:
            pass
        
        print("  ❌ CUDA nie znaleziona")
        print("  Pobierz CUDA Toolkit z: https://developer.nvidia.com/cuda-downloads")
        print("  Wybierz wersję kompatybilną z PyTorch")
    
    def configure_mps(self):
        """Konfiguruj MPS dla Apple Silicon"""
        if self.system != "Darwin" or self.machine != "arm64":
            print("ℹ️  MPS dostępne tylko na Apple Silicon Mac")
            return
        
        print("🍎 Konfiguracja Metal Performance Shaders...")
        
        # Test MPS w PyTorch
        test_code = """
import torch
if torch.backends.mps.is_available():
    print("  ✅ MPS dostępny w PyTorch")
    device = torch.device("mps")
    x = torch.randn(100, 100, device=device)
    print(f"  Test macierzy 100x100: OK")
else:
    print("  ❌ MPS niedostępny")
"""
        
        subprocess.run([sys.executable, "-c", test_code])
    
    def create_config_file(self):
        """Utwórz plik konfiguracyjny"""
        config = {
            "system": self.system,
            "machine": self.machine,
            "python_version": f"{self.python_version.major}.{self.python_version.minor}",
            "setup_date": str(Path.ctime(Path.cwd())),
            "gpu": self._check_gpu(),
            "paths": {
                "config_dir": str(self.config_dir),
                "venv": str(self.home / "ai_env")
            }
        }
        
        config_file = self.config_dir / "setup_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Konfiguracja zapisana w: {config_file}")
    
    def create_test_scripts(self):
        """Utwórz skrypty testowe"""
        tests_dir = Path("tests")
        tests_dir.mkdir(exist_ok=True)
        
        # Test GPU
        gpu_test = """#!/usr/bin/env python3
import torch
import platform

print(f"System: {platform.system()} {platform.machine()}")
print(f"PyTorch: {torch.__version__}")

# CUDA
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA not available")

# MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("MPS (Metal) available")
else:
    print("MPS not available")

# Test obliczeniowy
device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.matmul(x, y)
print(f"Matrix multiplication test on {device}: OK")
"""
        
        with open(tests_dir / "test_gpu.py", "w") as f:
            f.write(gpu_test)
        
        # Test MCP
        mcp_test = """#!/usr/bin/env node
console.log("Testing MCP setup...");

try {
    const { Server } = await import('@modelcontextprotocol/sdk/server/index.js');
    console.log("✅ MCP SDK imported successfully");
} catch (error) {
    console.log("❌ MCP SDK import failed:", error.message);
}
"""
        
        with open(tests_dir / "test_mcp.mjs", "w") as f:
            f.write(mcp_test)
        
        print("🧪 Skrypty testowe utworzone w katalogu 'tests'")


def main():
    parser = argparse.ArgumentParser(
        description="Konfiguracja środowiska AI/MCP"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Pełna instalacja wszystkich komponentów"
    )
    parser.add_argument(
        "--python-only", action="store_true",
        help="Tylko pakiety Python"
    )
    parser.add_argument(
        "--mcp-only", action="store_true",
        help="Tylko serwer MCP"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Uruchom testy po instalacji"
    )
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    print("🚀 Konfiguracja środowiska AI/MCP")
    print("=" * 50)
    
    # Informacje o systemie
    system_info = setup.check_system()
    print("\n📊 Informacje o systemie:")
    for key, value in system_info.items():
        if key == "gpu" and isinstance(value, dict):
            print(f"  {key}: {value.get('type', 'none')} "
                  f"({'available' if value.get('available') else 'not available'})")
            if value.get('details'):
                print(f"    Details: {value['details']}")
        else:
            print(f"  {key}: {value}")
    
    if args.full or (not args.python_only and not args.mcp_only):
        # Pełna instalacja
        print("\n🔨 Pełna instalacja...")
        
        # Python
        venv_path = setup.create_virtual_environment()
        
        # Pakiety Python
        packages = [
            "torch",
            "torchvision", 
            "torchaudio",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "pandas",
            "numpy",
            "matplotlib",
            "jupyter",
            "tensorboard"
        ]
        
        # Dodatkowe pakiety dla Windows
        if setup.system == "Windows" and system_info["gpu"]["type"] == "nvidia":
            packages.append("bitsandbytes-windows")
        elif setup.system == "Darwin":
            packages.append("mlx")
            packages.append("mlx-lm")
        
        setup.install_python_packages(packages)
        
        # Node.js i MCP
        setup.setup_nodejs()
        mcp_path = Path.cwd() / "mcp-server"
        setup.setup_mcp_server(mcp_path)
        
        # Konfiguracja GPU
        if system_info["gpu"]["type"] == "nvidia":
            setup.configure_cuda()
        elif system_info["gpu"]["type"] == "apple_silicon":
            setup.configure_mps()
    
    elif args.python_only:
        print("\n🐍 Instalacja pakietów Python...")
        packages = ["torch", "transformers", "datasets"]
        setup.install_python_packages(packages)
    
    elif args.mcp_only:
        print("\n🔧 Instalacja MCP...")
        setup.setup_nodejs()
        mcp_path = Path.cwd() / "mcp-server"
        setup.setup_mcp_server(mcp_path)
    
    # Utwórz pliki konfiguracyjne i testowe
    setup.create_config_file()
    setup.create_test_scripts()
    
    # Uruchom testy
    if args.test:
        print("\n🧪 Uruchamianie testów...")
        subprocess.run([sys.executable, "tests/test_gpu.py"])
        if Path("mcp-server").exists():
            subprocess.run(["node", "tests/test_mcp.mjs"])
    
    print("\n✅ Konfiguracja zakończona!")
    print("\n📝 Następne kroki:")
    print("1. Aktywuj środowisko wirtualne")
    print("2. Przejdź do przykładów w katalogu 'przyklady'")
    print("3. Uruchom test: python tests/test_gpu.py")


if __name__ == "__main__":
    main()