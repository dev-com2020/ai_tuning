# Ćwiczenia praktyczne

## Zasady pracy
- Pracujemy w parach. Każda runda to: zadanie → próba → samoocena → poprawka.
- W każdym ćwiczeniu określ kryteria akceptacji i format wyjścia.

---

## Moduł 2 — Prompt engineering

### Ćwiczenie 2.1: Usprawnianie promptu (iteracje)
- Zadanie: popraw jakość odpowiedzi dla krótkiego opisu produktu.
- Wejście: opis 3 funkcji, ograniczenie długości do 120 słów.
- Kryteria: brak halucynacji, styl neutralny, struktura: nagłówek + 3 punkty + CTA.
- Runda 1: napisz najprostszy prompt. Runda 2–3: iteracyjnie poprawiaj.

### Ćwiczenie 2.2: Wymuszanie formatu JSON
- Zadanie: zapytaj model o 3 ryzyka projektu i zwróć JSON zgodny ze schematem.
- Schemat kluczy: {"risk": string, "likelihood": oneOf[low,medium,high], "mitigation": string}.
- Walidacja: wynik musi parsować się bezbłędnie; brak pól spoza schematu.

### Ćwiczenie 2.3: Styl i ton
- Zadanie: ta sama treść w 2 stylach: „formalny” i „przystępny”.
- Kryteria: zgodność ze stylem, spójność faktów, identyczna zawartość merytoryczna.

---

## Moduł 3 — Jakość i bezpieczeństwo

### Ćwiczenie 3.1: Ograniczanie halucynacji
- Zadanie: zaprojektuj prompt, który wymusza: „odpowiedz TYLKO, jeśli masz pewność; inaczej przyznaj niepewność i zaproponuj 2 pytania doprecyzowujące”.
- Kryteria: brak wymyślonych faktów; obecność mechanizmu „odmowy”/eskalacji.

### Ćwiczenie 3.2: Moderacja treści
- Zadanie: przygotuj klasyfikator regułowy (prompt), który flaguje treści wg kategorii (np. przemoc, nienawiść, dane wrażliwe).
- Kryteria: jasne definicje kategorii, uzasadnienie decyzji, wynik w formacie JSON.

---

## Moduł 4 — Fine-tuning

### Ćwiczenie 4.1: Projekt danych (JSONL)
- Zadanie: zaprojektuj 10–20 przykładów instrukcja→odpowiedź dla jednego use-case’u firmowego.
- Kryteria: brak danych wrażliwych, zróżnicowanie przypadków, brak wycieków (data leakage).
- Przykład rekordu (JSONL):
```json
{"instruction": "Sformatuj opis zadania w JIRA wg szablonu.", "input": "Bug: Brak walidacji pola email...", "output": "[Tytuł] ...\n[Kroki] ...\n[Oczekiwane] ..."}
```

### Ćwiczenie 4.2: Walidacje zbioru
- Zadanie: policz rozkład etykiet/typów, wykryj duplikaty, oceń balans.
- Kryteria: raport z liczbami i rekomendacjami poprawek.

### Ćwiczenie 4.3: Uruchomienie procesu (suchy bieg)
- Zadanie: naszkicuj polecenia/konfigurację do uruchomienia SFT (np. HF TRL lub API).
- Kryteria: kompletność kroków: dane → config → trening → walidacja → eksport.
- Przykładowy szkic (pseudo):
```bash
# 1) Przygotuj env i dane
export DATA_PATH=./data/train.jsonl
# 2) Konfiguracja treningu (parametry, batch, LR)
# 3) Start treningu (np. trl SFTTrainer lub openai/fine_tuning)
# 4) Walidacja na dev set
# 5) Eksport i testy regresji
```

---

## Moduł 5 — Ewaluacja

### Ćwiczenie 5.1: Metryki i human eval
- Zadanie: zdefiniuj metryki dopasowane do celu oraz lekki proces human evaluation.
- Kryteria: metryki ilościowe + checklisty jakościowe; próg akceptacji.

---

## Moduł 6 — Zastosowania biznesowe

### Ćwiczenie 6.1: Mini-projekt wdrożeniowy
- Zadanie: opracuj szkic rozwiązania (architektura, dane, jakość, bezpieczeństwo, KPI).
- Kryteria: jasna odpowiedzialność komponentów, plan rollout, monitoring jakości.