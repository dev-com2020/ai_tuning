# Konspekt slajdów (deck outline)

## Deck 0 — Otwarcie i cele
- Tytuł szkolenia, agenda 2 dni
- Zasady pracy, oczekiwania, wyniki końcowe
- Ramy tematyczne: Moduły 1–6

## Deck 1 — Moduł 1: Zrozumienie LLM
- Architektura Transformera (self-attention, embedding, positional encoding)
- Linie modelowe: GPT, Gemini, Claude (różnice koncepcyjne)
- Ograniczenia i ryzyka: halucynacje, bias, kontekst, dryf
- Zastosowania i anty-zastosowania (gdzie nie używać)
- Slajd „Mapa mentalna LLM”

## Deck 2 — Moduł 2: Techniki efektywnego promptowania
- Zasady formułowania promptów: rola, kontekst, ograniczenia, format
- Zaawansowane techniki: system prompt, user prompt, chain-of-thought (omówienie), few-shot
- Wymuszanie stylu i tonu; instrukcje negatywne; stopniowanie złożoności
- Strukturyzacja wyników: JSON, YAML, schematy, walidacja
- Dobre i złe praktyki — przykłady porównawcze

## Deck 3 — Moduł 3: Kontrola jakości i bezpieczeństwa
- Źródła halucynacji i sposoby ograniczania (retrieval, walidacja, self-check)
- Sterowanie zakresem i stylem: zasady, ograniczenia, guardraile
- Moderacja: klasyfikacja treści, filtry, eskalacja
- Obwody bezpieczeństwa: wieloetapowe sprawdzanie odpowiedzi

## Deck 4 — Moduł 4: Fine-tuning LLM
- Kiedy stosować fine-tuning vs. prompt engineering/RAG
- Przygotowanie danych: formaty (JSONL/CSV), etyki, anonimizacja
- Przegląd narzędzi: OpenAI API, Hugging Face (TRL, PEFT, LoRA)
- Walidacje: rozkład etykiet, data leakage, sanity checks

## Deck 5 — Moduł 5: Ewaluacja jakości
- Metryki: perpleksja, BLEU, ROUGE, human-in-the-loop
- Testy regresji: zestawy przypadków, złote standardy
- A/B testy i panel ekspercki; monitoring po wdrożeniu

## Deck 6 — Moduł 6: Zastosowania biznesowe
- Chatboty i automatyzacja wsparcia
- Generowanie raportów i dokumentacji
- Personalizacja treści i rekomendacje
- Wzorce wdrożeń i kryteria sukcesu

## Deck X — Podsumowanie i plan wdrożenia
- Powtórka kluczowych zasad
- Lista kontrolna produkcyjna
- Ścieżki dalszego rozwoju zespołu