# Skrypt trenera (facilitation guide)

## Jak korzystać
- Każdy moduł zawiera: cele, sugestie demonstracji, aktywności, pytania kontrolne, ryzyka.
- Zakłada się elastyczność czasu vs. doświadczenie grupy (skracaj/rozszerzaj ćwiczenia).

---

## Dzień 1

### Moduł 1: Zrozumienie LLM (75 min)
- Cele:
  - Wyjaśnić architekturę Transformera i jej implikacje praktyczne.
  - Uświadomić ograniczenia (halucynacje, bias) i sposoby mitigacji.
- Demonstracje:
  - Krótka wizualizacja attention; pokaz różnic między modelami.
- Aktywności:
  - Dyskusja: gdzie w firmie LLM ma sens, a gdzie nie.
- Pytania kontrolne:
  - Co to jest self-attention? Jakie są typowe źródła halucynacji?
- Ryzyka:
  - Zbyt techniczne dygresje; pilnuj czasu.

### Moduł 2: Efektywne promptowanie (135 min)
- Cele:
  - Ustalić podstawy i zaawansowane wzorce formułowania promptów.
  - Nauczyć strukturyzować wyniki i wymuszać styl.
- Demonstracje:
  - Porównanie „złego” i „dobrego” promptu na tym samym zadaniu.
  - Wymuszenie formatu JSON i walidacja.
- Aktywności:
  - Iteracyjne usprawnianie promptu w parach (3 rundy).
  - Mini-lab: rozszerzanie kontekstu i kontrola tonu.
- Pytania kontrolne:
  - Jakie elementy powinien zawierać skuteczny prompt? Kiedy few-shot pomaga?
- Ryzyka:
  - Uczestnicy próbują „skrótów”; przypominaj o kryteriach akceptacji.

### Moduł 3: Jakość i bezpieczeństwo (90 min)
- Cele:
  - Ograniczać halucynacje; projektować guardraile i moderację.
- Demonstracje:
  - Pipeline: generacja → walidacja → moderacja → poprawka.
- Aktywności:
  - Ćwiczenie klasyfikacyjne: flagowanie wrażliwych treści.
- Pytania kontrolne:
  - Jak rozpoznać niepewność odpowiedzi? Co eskalować do człowieka?
- Ryzyka:
  - Nadmierna wiara w pojedynczy wynik modelu.

---

## Dzień 2

### Moduł 4: Fine-tuning (105 min)
- Cele:
  - Zrozumieć kiedy fine-tuning ma przewagę nad promptowaniem/RAG.
  - Przećwiczyć przygotowanie danych i uruchomienie procesu.
- Demonstracje:
  - Analiza przykładowego zbioru JSONL; sanity checks.
- Aktywności:
  - Projekt danych: 10–20 przykładów, definicja etykiet, split na zbiory.
- Pytania kontrolne:
  - Jakie są typowe pułapki (data leakage, nierównowaga klas)?
- Ryzyka:
  - Zbyt mało czasu na walidację — ogranicz zakres labu.

### Moduł 5: Ewaluacja (75 min)
- Cele:
  - Dobrać metryki do celu (BLEU/ROUGE vs. human eval).
- Demonstracje:
  - Porównanie modelu bazowego i dostrojonego na zestawie testowym.
- Aktywności:
  - Definicja kryteriów akceptacji i testów regresji.
- Pytania kontrolne:
  - Co mierzy perpleksja? Kiedy ufać ocenie człowieka?
- Ryzyka:
  - Mylenie metryk n-gramowych z jakością semantyczną.

### Moduł 6: Zastosowania biznesowe (75 min)
- Cele:
  - Przełożyć techniki na use-case’y: chatbot, raporty, personalizacja.
- Demonstracje:
  - Przykładowa architektura wdrożenia i monitoring jakości.
- Aktywności:
  - Mini-projekt: szkic rozwiązania dla własnego procesu.
- Pytania kontrolne:
  - Jak zdefiniować KPI wdrożenia? Jak planować rollout?
- Ryzyka:
  - Zbytnia ogólność — proś o konkretne procesy od zespołów.

---

## Materiały trenera
- Tablica (lub Miro), projektor, dostęp do notebooków/labu.
- Zestaw „złych” i „dobrych” promptów do porównań.
- Przykładowe dane do fine-tuningu (zanonimizowane, małe).

## Plan reagowania na ryzyka
- Gdy grupa zbyt techniczna: zwiększ udział labów, ogranicz prezentacje.
- Gdy grupa mniej techniczna: upraszczaj demo, wzmacniaj przykłady biznesowe.