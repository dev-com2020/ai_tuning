# SZCZEGÓŁOWA AGENDA SZKOLENIA: LLM I KONTROLOWANIE GENEROWANYCH ODPOWIEDZI

## Informacje ogólne
- **Czas trwania**: 2 dni (16 godzin szkoleniowych)
- **Forma**: Warsztaty praktyczne z elementami wykładu
- **Poziom**: Średnio-zaawansowany
- **Wymagania wstępne**: Podstawowa znajomość programowania, znajomość API REST

## DZIEŃ 1: Wprowadzenie do LLM i kontrolowanie generowanych odpowiedzi

### 9:00 - 9:15 | Powitanie i wprowadzenie
- Przedstawienie prowadzącego i uczestników
- Omówienie celów szkolenia
- Przedstawienie harmonogramu

### 9:15 - 10:45 | Moduł 1: Zrozumienie dużych modeli językowych
**Teoria (45 min)**
- Wprowadzenie do sztucznej inteligencji i NLP
- Architektury dużych modeli językowych:
  - Transformer - rewolucja w przetwarzaniu języka
  - GPT (Generative Pre-trained Transformer) - rodzina modeli OpenAI
  - Gemini - rozwiązania Google
  - Claude - modele Anthropic
- Zasada działania LLM:
  - Tokenizacja
  - Attention mechanism
  - Generowanie tekstu
- Kluczowe problemy:
  - Halucynacje - dlaczego modele "zmyślają"
  - Tendencyjność (bias) - skąd się bierze i jak ją minimalizować
  - Ograniczenia kontekstu

**Praktyka (45 min)**
- Demonstracja działania różnych modeli
- Porównanie odpowiedzi różnych LLM na te same pytania
- Identyfikacja halucynacji w praktyce

### 10:45 - 11:00 | Przerwa kawowa

### 11:00 - 13:00 | Moduł 2: Techniki tworzenia efektywnych promptów
**Teoria (60 min)**
- Podstawy prompt engineering:
  - Struktura promptu
  - Znaczenie kontekstu
  - Jasność i precyzja
- Zaawansowane techniki:
  - System prompt vs User prompt
  - Chain-of-thought (CoT) prompting
  - Few-shot learning
  - Zero-shot prompting
  - Role-playing w promptach
- Przykłady dobrych i złych praktyk

**Praktyka (60 min)**
- Warsztat: tworzenie promptów dla różnych przypadków użycia
- Analiza i optymalizacja promptów
- Praca w grupach nad złożonymi promptami

### 13:00 - 14:00 | Przerwa obiadowa

### 14:00 - 16:30 | Moduł 3: Kontrolowanie jakości i bezpieczeństwa generowanych treści
**Teoria (60 min)**
- Metody ograniczania ryzyka halucynacji:
  - Temperature i Top-p sampling
  - Frequency i Presence penalty
  - Kontekst i ograniczenia
- Kontrolowanie stylu i tonu:
  - Instrukcje stylystyczne
  - Przykłady w promptach
  - Persona modelu
- Implementacja zabezpieczeń:
  - Moderacja automatyczna
  - Filtrowanie treści
  - Walidacja odpowiedzi

**Praktyka (90 min)**
- Implementacja systemu kontroli jakości
- Tworzenie filtrów bezpieczeństwa
- Case study: bezpieczny chatbot korporacyjny

### 16:30 - 17:00 | Podsumowanie dnia pierwszego
- Q&A
- Zadania do samodzielnej pracy
- Zapowiedź dnia drugiego

## DZIEŃ 2: Fine-tuning i zaawansowane metody kontroli LLM

### 9:00 - 9:15 | Powitanie i podsumowanie dnia pierwszego
- Omówienie zadań domowych
- Pytania i wątpliwości

### 9:15 - 11:00 | Moduł 4: Fine-tuning dużych modeli językowych
**Teoria (45 min)**
- Czym jest fine-tuning?
  - Różnica między pre-training a fine-tuning
  - Kiedy stosować fine-tuning
  - Alternatywy: prompt engineering vs fine-tuning
- Przygotowanie danych:
  - Format danych treningowych
  - Czyszczenie i walidacja
  - Wielkość datasetu
- Proces dostosowywania:
  - OpenAI Fine-tuning API
  - Hugging Face Transformers
  - Parametry treningu

**Praktyka (60 min)**
- Przygotowanie datasetu do fine-tuningu
- Konfiguracja i uruchomienie procesu
- Monitorowanie treningu

### 11:00 - 11:15 | Przerwa kawowa

### 11:15 - 13:00 | Moduł 5: Metody oceny jakości modeli językowych
**Teoria (45 min)**
- Metryki automatyczne:
  - Perpleksja - miara niepewności modelu
  - BLEU - ocena podobieństwa tłumaczeń
  - ROUGE - ocena jakości streszczeń
  - BERTScore - semantyczna ocena podobieństwa
- Human Evaluation:
  - Metodologie oceny przez ludzi
  - Skale oceny
  - Inter-rater agreement
- Ocena fine-tuningu:
  - Overfitting vs underfitting
  - Validation loss
  - A/B testing

**Praktyka (60 min)**
- Implementacja systemu ewaluacji
- Porównanie modeli przed i po fine-tuningu
- Tworzenie dashboardu metryk

### 13:00 - 14:00 | Przerwa obiadowa

### 14:00 - 16:00 | Moduł 6: Praktyczne zastosowania kontrolowania LLM w biznesie
**Case Study 1: Chatboty i obsługa klienta (40 min)**
- Architektura systemu
- Integracja z CRM
- Obsługa wielojęzyczna
- Eskalacja do człowieka

**Case Study 2: Generowanie dokumentacji (40 min)**
- Automatyczne raporty
- Analizy biznesowe
- Dokumentacja techniczna
- Kontrola jakości i zgodności

**Case Study 3: Personalizacja treści (40 min)**
- Profilowanie użytkowników
- Dynamiczne dostosowanie contentu
- A/B testing treści
- Mierzenie efektywności

### 16:00 - 17:00 | Projekt końcowy i podsumowanie
- Praca w grupach nad mini-projektem
- Prezentacja rozwiązań
- Dyskusja i feedback
- Certyfikaty ukończenia

## Materiały dodatkowe
- Prezentacje w formacie PDF
- Kod źródłowy przykładów
- Lista zalecanych lektur
- Dostęp do platformy e-learningowej (3 miesiące)
- Grupa wsparcia na Slack

## Wymagania techniczne
- Laptop z dostępem do internetu
- Konto w OpenAI API (opcjonalnie)
- Środowisko Python 3.8+
- Edytor kodu (VS Code zalecany)