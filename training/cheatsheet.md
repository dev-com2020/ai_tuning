# Ściąga: LLM w praktyce

## Skuteczny prompt — checklista
- Rola/persona: „Jesteś… (ekspert, asystent…)”.
- Kontekst i cel: co, dla kogo, dlaczego, ograniczenia.
- Format wyjścia: np. JSON/YAML/markdown; walidowalny schemat.
- Styl i ton: formalny/przystępny/neutralny; długość; język.
- Kryteria akceptacji: poprawność faktów, brak halucynacji, test walidatora.
- Iteracje: poproś o samokontrolę (self-check) i poprawkę.

## Ograniczanie halucynacji
- Wymuś „odmowę” przy braku pewności + pytania doprecyzowujące.
- Dodaj retrieval/kontekst źródłowy; linkuj źródła.
- Waliduj fakty: reguły, klasyfikatory, cross-check.

## Fine-tuning — kiedy warto
- Potrzeba stałego stylu/domeny lub nowych umiejętności.
- Gdy prompt/RAG nie wystarcza dla jakości/KPI.

## Ewaluacja
- Ilościowe: perpleksja, BLEU, ROUGE.
- Jakościowe: human evaluation, checklisty, A/B.
- Testy regresji na złotych przykładach.

## Moderacja i bezpieczeństwo
- Klasyfikacja treści: kategorie, progi, eskalacja do człowieka.
- Ochrona danych: anonimizacja, logowanie, retencja.

## Wymuszanie formatu JSON (wzorzec)
```text
Zwróć WYŁĄCZNIE JSON bez komentarzy i tekstu. Schemat:
{
  "risk": string,
  "likelihood": "low"|"medium"|"high",
  "mitigation": string
}
Jeśli niepewne — zwróć: {"error": "insufficient_context"}.
```