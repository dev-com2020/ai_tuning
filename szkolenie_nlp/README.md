# Szkolenie: Przetwarzanie Języka Naturalnego (NLP) 🚀

## 📚 O szkoleniu

Dwudniowe intensywne szkolenie z zakresu Natural Language Processing (NLP) - od podstaw do zaawansowanych zastosowań biznesowych.

### Dzień 1: Wprowadzenie do NLP i podstawowe techniki
- **Moduł 1**: Wstęp do NLP i jego zastosowań
- **Moduł 2**: Narzędzia i biblioteki (NLTK, spaCy, Hugging Face, OpenAI)
- **Moduł 3**: Podstawowe operacje (tokenizacja, lematyzacja, POS tagging)
- **Warsztaty**: Praktyczne projekty (analiza sentymentu, klasyfikacja tekstu)

### Dzień 2: Zaawansowane modele NLP i zastosowania biznesowe
- **Moduł 4**: Nowoczesne modele (Transformery, BERT, GPT, T5)
- **Moduł 5**: Generowanie i rozumienie tekstu
- **Moduł 6**: NLP w biznesie (chatboty, automatyzacja dokumentów)

---

## 🔧 Instalacja i konfiguracja

### 1. Wymagania systemowe
- Python 3.8 lub nowszy
- Co najmniej 8GB RAM
- ~5GB wolnego miejsca na dysku (dla modeli)
- (Opcjonalnie) GPU z CUDA dla szybszego treningu

### 2. Instalacja środowiska

#### Opcja A: Conda (zalecane)
```bash
# Utwórz nowe środowisko
conda create -n nlp-training python=3.10
conda activate nlp-training

# Zainstaluj zależności
pip install -r requirements.txt
```

#### Opcja B: venv
```bash
# Utwórz wirtualne środowisko
python -m venv venv

# Aktywuj (Windows)
venv\Scripts\activate

# Aktywuj (Linux/Mac)
source venv/bin/activate

# Zainstaluj zależności
pip install -r requirements.txt
```

### 3. Pobierz modele językowe

#### spaCy
```bash
# Model polski
python -m spacy download pl_core_news_sm

# Model angielski
python -m spacy download en_core_web_sm

# (Opcjonalnie) Większy model z wektorami
python -m spacy download en_core_web_md
```

#### NLTK
```python
# Uruchom Python i wykonaj:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Konfiguracja OpenAI API (opcjonalne)

Jeśli chcesz korzystać z OpenAI API:

```bash
# Ustaw zmienną środowiskową (Linux/Mac)
export OPENAI_API_KEY='your-api-key-here'

# Windows (PowerShell)
$env:OPENAI_API_KEY='your-api-key-here'

# Lub w pliku .env
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Klucz API można uzyskać na: https://platform.openai.com/

---

## 📁 Struktura projektu

```
szkolenie_nlp/
│
├── dzien1/                          # Materiały Dzień 1
│   ├── modul1_wprowadzenie_do_nlp.ipynb
│   ├── modul2_narzedzia_biblioteki.ipynb
│   ├── modul3_podstawowe_operacje.ipynb
│   └── warsztaty_dzien1.ipynb
│
├── dzien2/                          # Materiały Dzień 2
│   ├── modul4_transformery_modele.ipynb
│   ├── modul5_generowanie_rozumienie.ipynb
│   └── modul6_nlp_biznes.ipynb
│
├── dane/                            # Przykładowe dane
│   ├── sample_reviews.csv
│   ├── sample_documents.csv
│   └── sample_emails.txt
│
├── resources/                       # Dodatkowe zasoby
│
├── requirements.txt                 # Zależności
└── README.md                        # Ten plik
```

---

## 🚀 Jak korzystać z materiałów

### Uruchomienie Jupyter Notebook

```bash
# Aktywuj środowisko
conda activate nlp-training  # lub: source venv/bin/activate

# Uruchom Jupyter
jupyter notebook

# Jupyter otworzy się w przeglądarce
# Przejdź do katalogu dzien1/ lub dzien2/ i otwórz notebooki
```

### Kolejność nauki

1. **Dzień 1**
   - Zacznij od `modul1_wprowadzenie_do_nlp.ipynb`
   - Przejdź kolejno przez moduły 2 i 3
   - Zakończ warsztatami praktycznymi

2. **Dzień 2**
   - Kontynuuj od `modul4_transformery_modele.ipynb`
   - Następnie moduły 5 i 6
   - Wykonaj projekty z modułu 6

### Wskazówki

- ✅ Wykonuj kod w komórkach krok po kroku
- ✅ Eksperymentuj z parametrami
- ✅ Próbuj własnych przykładów
- ✅ Zadawaj pytania
- ✅ Rób notatki

---

## 📊 Przykładowe dane

W katalogu `dane/` znajdziesz:

- **sample_reviews.csv** - recenzje produktów (PL/EN) z ocenami
- **sample_documents.csv** - artykuły z różnych kategorii
- **sample_emails.txt** - przykładowe emaile biznesowe

Możesz używać tych danych do testowania i eksperymentów.

---

## 🔍 Rozwiązywanie problemów

### Problem: Model spaCy nie został znaleziony
```bash
# Zainstaluj model ponownie
python -m spacy download pl_core_news_sm
```

### Problem: Brak modułu transformers
```bash
pip install --upgrade transformers
```

### Problem: Out of Memory podczas ładowania modelu
```python
# Użyj mniejszego modelu:
# Zamiast: model="bert-base-uncased"
# Użyj: model="distilbert-base-uncased"
```

### Problem: Wolne działanie
- Zmniejsz `batch_size`
- Użyj mniejszych modeli (np. DistilBERT zamiast BERT)
- Ogranicz `max_length` w tokenizacji
- Rozważ użycie GPU

### Problem: OpenAI API zwraca błąd
- Sprawdź czy masz ustawiony klucz API
- Sprawdź limity swojego konta
- Upewnij się, że masz aktywną subskrypcję

---

## 📚 Dodatkowe zasoby

### Dokumentacja
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [spaCy](https://spacy.io/usage)
- [NLTK](https://www.nltk.org/)
- [OpenAI API](https://platform.openai.com/docs/)

### Kursy online
- [Hugging Face Course](https://huggingface.co/course) - darmowy kurs NLP
- [Fast.ai NLP](https://www.fast.ai/) - praktyczne podejście
- [DeepLearning.AI NLP Specialization](https://www.deeplearning.ai/)

### Community
- [Hugging Face Discord](https://discord.com/invite/JfAtkvEtRb)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [r/LanguageTechnology](https://www.reddit.com/r/LanguageTechnology/)

### Artykuły i Papers
- [Papers With Code - NLP](https://paperswithcode.com/area/natural-language-processing)
- [arXiv - Computation and Language](https://arxiv.org/list/cs.CL/recent)

### Modele i Datasety
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Hugging Face Datasets](https://huggingface.co/datasets)

---

## 💡 Projekty do samodzielnej praktyki

Po ukończeniu szkolenia, wypróbuj te projekty:

1. **Chatbot FAQ**
   - Stwórz bota odpowiadającego na często zadawane pytania
   - Użyj intencji i ekstrakcji encji

2. **System rekomendacji treści**
   - Analizuj preferencje użytkowników
   - Rekomenduj podobne artykuły/produkty

3. **Automatyczne tagowanie treści**
   - Klasyfikuj artykuły według kategorii
   - Generuj tagi/słowa kluczowe

4. **Analiza opinii o produkcie**
   - Zbieraj recenzje z różnych źródeł
   - Analizuj sentyment i aspekty (ABSA)

5. **System Q&A dla dokumentacji**
   - Zbuduj wyszukiwarkę w dokumentach firmy
   - Implementuj question answering

---

## 🤝 Wsparcie

Jeśli masz pytania lub problemy:

1. Sprawdź sekcję "Rozwiązywanie problemów" powyżej
2. Przeszukaj dokumentację bibliotek
3. Zadaj pytanie prowadzącemu szkolenie
4. Sprawdź community (Discord, Reddit)

---

## 📝 Certyfikat

Po ukończeniu szkolenia i wykonaniu wszystkich modułów otrzymasz certyfikat potwierdzający udział w szkoleniu NLP.

---

## ⚖️ Licencja

Materiały szkoleniowe są dostępne wyłącznie dla uczestników szkolenia.

---

## 🎯 Następne kroki

Po szkoleniu:

1. ✅ Przejrzyj wszystkie notebooki jeszcze raz
2. ✅ Wykonaj dodatkowe ćwiczenia
3. ✅ Zbuduj własny projekt
4. ✅ Podziel się wiedzą z zespołem
5. ✅ Śledź najnowsze trendy w NLP

---

**Powodzenia w nauce NLP! 🚀**

*Ostatnia aktualizacja: Październik 2024*
