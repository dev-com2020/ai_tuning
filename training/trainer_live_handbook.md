# Podręcznik prowadzenia na żywo (minute‑by‑minute)

Cel: wesprzeć prowadzącego w trakcie zajęć. Zawiera scenariusze minutowe, skrypty przejść, przykładowe wypowiedzi, checkpointy jakości i timeboxy.

Legenda: [S] — skrypt mówiony, [A] — aktywność, [C] — checkpoint, [T] — tip.

---

## Dzień 1 (8h)

### Otwarcie (09:30–10:00)
- 09:30–09:33 [S]: „Witajcie! Dzisiejszy cel: skuteczne prompty i kontrola jakości…”
- 09:33–09:36 [A]: Oczekiwania na sticky notes (1 per osoba).
- 09:36–09:40 [S]: Omówienie modułów i zasad (pytania na bieżąco, parking lot).
- 09:40–09:45 [A]: Szybka rundka: imię, rola, 1 oczekiwanie.
- 09:45–10:00 [S]: „Mapa szkolenia” — slajdy Deck 0.
- [C]: Czy widać slajdy? Czy wszyscy mają dostęp do materiałów?

### Moduł 1: Zrozumienie LLM (10:00–11:15)
- 10:00–10:05 [S]: „Transformer: klucz — self‑attention. Co to zmienia praktycznie?”
- 10:05–10:15 [S]: Ograniczenia: halucynacje, bias, kontekst.
- 10:15–10:25 [A]: Dyskusja: gdzie LLM ma sens vs. nie?
- 10:25–10:35 [S]: Linie modelowe (GPT/Gemini/Claude) — różnice koncepcyjne.
- 10:35–10:45 [A]: Mini‑case: „kiedy zaufać, kiedy eskalować do człowieka”.
- 10:45–11:10 [A]: Praca w parach: wypisz 3 kryteria jakości dla waszego use‑case’u.
- 11:10–11:15 [C]: Zbiórka wniosków, 3 bullet‑y na tablicy.

[Przerwa 11:15–11:30]

### Moduł 2: Prompt engineering (11:30–13:45)
- 11:30–11:35 [S]: „Prompt = rola + kontekst + format + styl + kryteria. Zaczynamy.”
- 11:35–12:00 [A]: Ćw. 2.1 — iteracje promptu (solo→para). Timebox: 15 + 10 min.
- 12:00–12:10 [C]: Demo porównań „zły vs dobry” prompt.
- 12:10–12:30 [A]: Ćw. 2.2 — wymuszanie JSON + walidacja `jq`.
- 12:30–12:45 [S]: Zaawansowane techniki: system prompt, few‑shot, CoT (omówienie).

[Przerwa obiadowa 12:45–13:45]

- 13:45–14:15 [A]: Ćw. 2.3 — styl i ton: formalny vs przystępny.
- 14:15–14:35 [A]: Mini‑lab: strukturyzacja odpowiedzi pod schemat (JSON/YAML).
- 14:35–14:45 [C]: Check jakości: brak halucynacji, zgodność ze schematem, długość.

[Przerwa 15:00–15:15]

### Moduł 3: Jakość i bezpieczeństwo (15:15–16:45)
- 15:15–15:25 [S]: „Skąd halucynacje? Jak je ograniczać (retrieval, self‑check)?”
- 15:25–15:45 [A]: Ćw. 3.1 — mechanizm odmowy przy niepewności + pytania.
- 15:45–16:05 [A]: Ćw. 3.2 — moderacja: klasyfikator regułowy z JSON output.
- 16:05–16:25 [S]: Obwody bezpieczeństwa i eskalacja do człowieka.
- 16:25–16:45 [C]: Przegląd wyników: poprawność, bezpieczeństwo, wytyczne.

### Zamknięcie dnia (16:45–17:00)
- 16:45–16:55 [A]: Exit ticket: 1 insight, 1 pytanie.
- 16:55–17:00 [S]: „Jutro: fine‑tuning, ewaluacja, use‑case’y. Dzięki!”

---

## Dzień 2 (8h)

### Rozgrzewka (09:30–10:00)
- 09:30–09:40 [S]: Rekap dnia 1, wnioski z ticketów.
- 09:40–10:00 [A]: Konfiguracja środowiska labowego (pary, check `.env`).

### Moduł 4: Fine‑tuning (10:00–12:30)
- 10:00–10:10 [S]: „Kiedy SFT, a kiedy wystarczy prompt/RAG?”
- 10:10–10:35 [A]: Ćw. 4.1 — projekt danych (10–20 przykładów JSONL).
- 10:35–11:00 [A]: Ćw. 4.2 — walidacje zbioru (balans, duplikaty, leakage).
- 11:00–11:15 [S]: Narzędzia: OpenAI/HF, TRL, PEFT/LoRA — przegląd.
- 11:15–11:30 [Przerwa]
- 11:30–12:30 [A]: Ćw. 4.3 — suchy bieg procesu SFT (kroki, config, testy).
- [C]: Czy kroki: dane→config→trening→walidacja→eksport są kompletne?

[Przerwa obiadowa 12:30–13:30]

### Moduł 5: Ewaluacja (13:30–14:45)
- 13:30–13:40 [S]: Metryki: perpleksja, BLEU, ROUGE vs human eval.
- 13:40–14:10 [A]: Definicja testów regresji i progów akceptacji.
- 14:10–14:45 [C]: Porównanie bazowy vs dostrojony: wnioski i decyzje.

[Przerwa 14:45–15:00]

### Moduł 6: Zastosowania biznesowe (15:00–16:15)
- 15:00–15:10 [S]: „Od problemu do KPI: jak ocenić wartość przypadku użycia”.
- 15:10–15:55 [A]: Mini‑projekt zespołowy: szkic rozwiązania i plan rollout.
- 15:55–16:15 [C]: Prezentacje 1‑min + feedback, plan po szkoleniu.

### Finał (16:15–17:00)
- 16:15–16:35 [A]: Quiz (`training/quiz.md`) i szybkie omówienie.
- 16:35–16:55 [S]: Podsumowanie i lista kontrolna wdrożenia.
- 16:55–17:00 [S]: Materiały i kanał wsparcia po szkoleniu.

---

## Przejścia i skrypty (gotowce)
- Do dyskusji: „Zatrzymajmy się. Co by się stało, gdyby…?”
- Do ćwiczeń: „Najpierw solo 5 min, potem para 10 min, na końcu 5 min na wnioski.”
- Do cięcia dygresji: „Zanotujmy to na parkingu i wróćmy w Q&A.”
- Do energii: „Wstańmy na 30 sekund, zamiana miejsc i jedziemy dalej.”

## Checkpointy jakości
- Prompt: ma rolę, kontekst, format, styl, kryteria? Jest test JSON?
- Odpowiedź: brak halucynacji, zgodność z ograniczeniami i schematem, długość OK.
- SFT: dane zbalansowane, bez duplikatów i leakage; config spójny z celem.
- Ewaluacja: metryki + human eval; próg akceptacji i test regresji.

## Lista potrzeb do modułów (skrót)
- M2: `jq`/`jsonlint`, przykłady złych/dobrych promptów.
- M4: próbki danych JSONL, template config.
- M5: zestaw „złotych” przykładów i baseline outputs.

## Notatki prowadzącego
- Miejsce na decyzje ad hoc, problemy sali, sugestie ulepszeń.