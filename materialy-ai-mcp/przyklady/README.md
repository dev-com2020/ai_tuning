# Przykłady kodu AI i MCP

Ten katalog zawiera praktyczne przykłady implementacji AI i MCP dla różnych zastosowań.

## 📂 Struktura

```
przyklady/
├── fine-tuning-examples.py    # Przykłady fine-tuningu modeli
├── mcp-server-example.js      # Przykładowy serwer MCP
├── README.md                  # Ten plik
└── projects/                  # Przykładowe projekty
    ├── chatbot/              # Chatbot z AI
    ├── code-assistant/       # Asystent programisty
    └── data-analyzer/        # Analizator danych
```

## 🎯 Przykłady fine-tuningu (fine-tuning-examples.py)

### 1. Podstawowy fine-tuning
```python
from przyklady.fine_tuning_examples import UniversalFineTuner

# Inicjalizacja
tuner = UniversalFineTuner("microsoft/phi-2")

# Załaduj model z LoRA
tuner.load_model(use_lora=True)

# Przygotuj dane
dataset = tuner.prepare_dataset("data.json", format_type="instruction")

# Trenuj
tuner.train(dataset, epochs=3)
```

### 2. Fine-tuning dla różnych zadań

#### Klasyfikacja tekstu
```python
# Sentiment analysis
sentiment_data = [
    {"text": "Świetny produkt!", "label": 1},
    {"text": "Bardzo słaba jakość", "label": 0}
]

# Model BERT dla polskiego
model_name = "allegro/herbert-base-cased"
# ... zobacz przykład w pliku
```

#### Generowanie kodu
```python
# Dataset z przykładami kodu
code_data = [
    {
        "instruction": "Napisz funkcję sortującą",
        "input": "lista = [3, 1, 4, 1, 5]",
        "output": "def sort_list(lst):\n    return sorted(lst)"
    }
]
```

## 🔧 Przykłady MCP (mcp-server-example.js)

### Podstawowy serwer MCP
```javascript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';

const server = new Server({
  name: 'my-server',
  version: '1.0.0'
});

// Dodaj narzędzie
server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'hello',
    description: 'Przywitaj się',
    inputSchema: { /* ... */ }
  }]
}));
```

### Zaawansowane funkcje MCP

#### 1. Analiza plików
```javascript
// Narzędzie do analizy kodu
{
  name: 'analyze_file',
  description: 'Analizuje plik i zwraca statystyki',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string' },
      detailed: { type: 'boolean' }
    }
  }
}
```

#### 2. Wyszukiwanie
```javascript
// Wyszukiwanie w plikach
{
  name: 'search_files',
  description: 'Wyszukuje pliki według wzorca',
  inputSchema: {
    type: 'object',
    properties: {
      pattern: { type: 'string' },
      directory: { type: 'string' }
    }
  }
}
```

## 🚀 Przykładowe projekty

### 1. Chatbot AI
```bash
cd projects/chatbot
npm install
npm start
```

Funkcje:
- Konwersacja z użytkownikiem
- Pamięć kontekstu
- Integracja z MCP dla dostępu do plików

### 2. Asystent programisty
```bash
cd projects/code-assistant
python main.py
```

Funkcje:
- Generowanie kodu
- Refaktoryzacja
- Dokumentacja
- Analiza błędów

### 3. Analizator danych
```bash
cd projects/data-analyzer
python analyze.py --input data.csv
```

Funkcje:
- Analiza statystyczna
- Wizualizacje
- Predykcje ML
- Raporty

## 💡 Najlepsze praktyki

### Fine-tuning
1. **Zacznij od małego**: Testuj na małym datasecie
2. **Monitoruj metryki**: Loss, accuracy, overfitting
3. **Używaj LoRA**: Dla efektywności pamięci
4. **Walidacja**: Zawsze miej zbiór walidacyjny

### MCP
1. **Bezpieczeństwo**: Waliduj wszystkie wejścia
2. **Wydajność**: Cache'uj wyniki
3. **Błędy**: Obsługuj gracefully
4. **Dokumentacja**: Opisuj wszystkie narzędzia

## 📊 Porównanie wydajności

| Model | Zadanie | GPU | Czas treningu | Accuracy |
|-------|---------|-----|---------------|----------|
| Phi-2 | Instrukcje | RTX 3090 | 2h | 89% |
| LLaMA-7B | Chat | A100 | 8h | 92% |
| BERT | Klasyfikacja | M1 Max | 1h | 94% |

## 🛠️ Troubleshooting

### Problem: Out of Memory
```python
# Rozwiązanie 1: Zmniejsz batch size
training_args.per_device_train_batch_size = 1

# Rozwiązanie 2: Użyj gradient accumulation
training_args.gradient_accumulation_steps = 16

# Rozwiązanie 3: Kwantyzacja
model = load_in_4bit=True
```

### Problem: Wolny trening
```python
# Użyj mixed precision
training_args.fp16 = True  # lub bf16

# Więcej workerów
dataloader_num_workers = 4

# Kompilacja modelu (PyTorch 2.0+)
model = torch.compile(model)
```

## 📚 Dodatkowe zasoby

### Tutoriale
1. [Fine-tuning krok po kroku](../fine-tuning/windows/przewodnik-windows.md)
2. [MCP od podstaw](../wprowadzenie/podstawy-ai-mcp.md)
3. [Optymalizacja dla Mac](../fine-tuning/mac/przewodnik-mac.md)

### Datasety
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Polish NLP Resources](https://github.com/topics/polish-nlp)
- [Code datasets](https://huggingface.co/datasets?task_categories=task_categories:text-generation&tags=code)

### Modele
- [Małe modele (<3B)](https://huggingface.co/models?other=base_model:size_categories:0-3B)
- [Modele dla polskiego](https://huggingface.co/models?language=pl)
- [Modele do kodu](https://huggingface.co/models?pipeline_tag=text-generation&tags=code)

## 🎮 Interaktywne demo

Uruchom Jupyter Notebook z przykładami:
```bash
jupyter notebook interactive_examples.ipynb
```

Zawiera:
- Interaktywny fine-tuning
- Wizualizacja procesu uczenia
- Porównanie modeli
- Playground MCP

## 🤝 Współpraca

Masz ciekawy przykład? Prześlij PR!

1. Fork repozytorium
2. Dodaj przykład do odpowiedniego katalogu
3. Zaktualizuj README
4. Stwórz Pull Request

---

Szczęśliwego kodowania! 🚀