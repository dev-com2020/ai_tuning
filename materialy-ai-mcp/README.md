# 🚀 Materiały edukacyjne AI i MCP

Kompletny zestaw materiałów do nauki o sztucznej inteligencji (AI) i Model Context Protocol (MCP), z przewodnikami fine-tuningu dla systemów Windows i macOS.

## 📚 Spis treści

### 1. [Wprowadzenie](./wprowadzenie/podstawy-ai-mcp.md)
- Czym jest AI i Machine Learning
- Model Context Protocol (MCP) - nowy standard integracji
- Architektura i zastosowania
- Pierwsze kroki

### 2. Fine-tuning modeli AI
- **[🪟 Przewodnik dla Windows](./fine-tuning/windows/przewodnik-windows.md)**
  - Konfiguracja środowiska z CUDA
  - Optymalizacja dla GPU NVIDIA
  - Rozwiązywanie problemów Windows-specific
  
- **[🍎 Przewodnik dla macOS](./fine-tuning/mac/przewodnik-mac.md)**
  - Wykorzystanie Apple Silicon i MPS
  - Framework MLX od Apple
  - Optymalizacja dla architektury ARM

### 3. [Przykłady kodu](./przyklady/)
- **[fine-tuning-examples.py](./przyklady/fine-tuning-examples.py)** - Uniwersalne przykłady fine-tuningu
- **[mcp-server-example.js](./przyklady/mcp-server-example.js)** - Przykładowy serwer MCP z różnymi funkcjonalnościami

### 4. [Narzędzia pomocnicze](./narzedzia/)
- **[setup-environment.py](./narzedzia/setup-environment.py)** - Automatyczna konfiguracja środowiska
- Skrypty benchmarkowe i monitorujące
- Konwertery modeli

## 🎯 Dla kogo są te materiały?

- **Programiści** chcący rozszerzyć swoje aplikacje o AI
- **Data Scientists** szukający praktycznych przykładów
- **Studenci** uczący się o najnowszych technologiach AI
- **Entuzjaści** eksplorujący możliwości AI i MCP

## 🛠️ Wymagania

### Minimalne wymagania sprzętowe:
- **CPU**: Intel i5/AMD Ryzen 5 lub Apple M1
- **RAM**: 16GB (zalecane 32GB+)
- **Dysk**: 100GB wolnego miejsca
- **GPU** (opcjonalne): NVIDIA z 8GB+ VRAM lub Apple Silicon

### Wymagania programowe:
- **Python**: 3.8 - 3.11
- **Node.js**: 18.0+
- **System**: Windows 10+, macOS 10.15+, lub Linux

## 🚀 Szybki start

### 1. Sklonuj lub pobierz materiały
```bash
git clone <repository-url>
cd materialy-ai-mcp
```

### 2. Uruchom skrypt konfiguracyjny
```bash
# Pełna instalacja
python narzedzia/setup-environment.py --full

# Tylko dla Pythona
python narzedzia/setup-environment.py --python-only

# Tylko dla MCP
python narzedzia/setup-environment.py --mcp-only
```

### 3. Sprawdź przykłady
```bash
# Fine-tuning
python przyklady/fine-tuning-examples.py --example 1

# Serwer MCP
cd przyklady
npm install
node mcp-server-example.js
```

## 📖 Ścieżka nauki

### Początkujący (1-2 tygodnie)
1. Przeczytaj [wprowadzenie do AI i MCP](./wprowadzenie/podstawy-ai-mcp.md)
2. Skonfiguruj środowisko używając narzędzi
3. Uruchom przykład fine-tuningu małego modelu
4. Stwórz prosty serwer MCP

### Średniozaawansowany (2-4 tygodnie)
1. Przestudiuj przewodnik fine-tuningu dla swojego systemu
2. Eksperymentuj z różnymi modelami i datasetami
3. Rozbuduj serwer MCP o własne funkcjonalności
4. Zintegruj AI z istniejącą aplikacją

### Zaawansowany (1+ miesiąc)
1. Optymalizuj modele dla produkcji
2. Implementuj zaawansowane techniki (LoRA, QLoRA)
3. Stwórz własny framework MCP
4. Wdróż rozwiązanie w chmurze

## 💡 Najważniejsze koncepty

### AI i Machine Learning
- **Fine-tuning**: Dostosowywanie pre-trenowanych modeli do konkretnych zadań
- **LoRA**: Efektywna metoda fine-tuningu dużych modeli
- **Quantization**: Redukcja rozmiaru modelu z minimalną utratą jakości
- **Prompt Engineering**: Sztuka tworzenia efektywnych zapytań do AI

### Model Context Protocol (MCP)
- **Serwery**: Udostępniają zasoby i funkcjonalności
- **Klienci**: Aplikacje AI korzystające z zasobów
- **Bezpieczeństwo**: Kontrolowany dostęp do lokalnych zasobów
- **Standaryzacja**: Jednolity sposób integracji

## 🔥 Popularne przypadki użycia

### 1. Asystent kodowania
- Fine-tuning modelu na własnym kodzie
- Serwer MCP z dostępem do dokumentacji
- Integracja z IDE

### 2. Chatbot firmowy
- Model wytrenowany na danych firmowych
- MCP dla dostępu do baz danych
- Interfejs webowy

### 3. Analizator dokumentów
- Model do ekstrakcji informacji
- MCP do zarządzania plikami
- Automatyzacja procesów

## 📊 Porównanie platform

| Cecha | Windows (NVIDIA) | macOS (Apple Silicon) | Linux |
|-------|-----------------|----------------------|--------|
| Wydajność GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Łatwość konfiguracji | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Wsparcie modeli | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Efektywność energetyczna | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Koszt | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

## 🐛 Rozwiązywanie problemów

### Najczęstsze problemy:

1. **Out of Memory (OOM)**
   - Zmniejsz batch size
   - Użyj gradient accumulation
   - Włącz kwantyzację

2. **Wolny trening**
   - Sprawdź wykorzystanie GPU
   - Użyj mixed precision training
   - Zoptymalizuj data loading

3. **Błędy importu**
   - Sprawdź wersje pakietów
   - Użyj odpowiedniego środowiska wirtualnego
   - Zainstaluj brakujące zależności

## 🤝 Społeczność i wsparcie

### Gdzie szukać pomocy:
- **Discord**: [AI Developers Polska](https://discord.gg/...)
- **Forum**: [Hugging Face Forums](https://discuss.huggingface.co/)
- **Stack Overflow**: Tag `transformers`, `mcp`
- **GitHub Issues**: W odpowiednich repozytoriach

### Przydatne zasoby:
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MCP Specification](https://modelcontextprotocol.io)
- [Papers with Code](https://paperswithcode.com/)

## 📈 Dalszy rozwój

### Co dalej?
1. **Eksploruj nowe modele**: LLaMA 3, Mistral, Gemma
2. **Testuj różne techniki**: DPO, RLHF, Constitutional AI
3. **Buduj własne narzędzia**: Rozszerzenia MCP, custom trainers
4. **Dziel się wiedzą**: Pisz blogi, twórz tutoriale

### Trendy 2024/2025:
- 🔮 Multimodalne modele (tekst + obraz + dźwięk)
- 🎯 Specialized fine-tuning dla konkretnych branż
- 🔐 Privacy-preserving AI
- ⚡ Edge AI i modele na urządzeniach
- 🌐 Federacyjne uczenie

## 📝 Licencja i uznania

Te materiały są udostępnione na licencji MIT. Możesz je swobodnie używać, modyfikować i rozpowszechniać.

### Podziękowania:
- Społeczność Hugging Face za niesamowite narzędzia
- Anthropic za Model Context Protocol
- Wszystkim kontrybutorom open-source AI

---

**Ostatnia aktualizacja**: Październik 2025

**Wersja**: 1.0.0

**Autor**: AI Learning Community

---

🌟 **Powodzenia w Twojej przygodzie z AI!** 🌟