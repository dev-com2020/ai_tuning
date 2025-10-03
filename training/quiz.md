# Quiz sprawdzający wiedzę

Instrukcja: Odpowiedz krótko. Dla pytań A–D wybierz jedną odpowiedź.

## Pytania
1. (A–D) Co jest kluczowym mechanizmem w architekturze Transformera?
   - A: Convolution
   - B: Self-attention
   - C: Recurrent connections
   - D: Pooling
2. Podaj dwie typowe przyczyny halucynacji w LLM.
3. (A–D) Który element NIE należy do dobrego promptu?
   - A: Rola/Persona
   - B: Kontekst i ograniczenia
   - C: Losowe emoji dla urozmaicenia
   - D: Kryteria akceptacji
4. Wyjaśnij, kiedy warto użyć few-shot prompting.
5. (A–D) Co oznacza perpleksja?
   - A: Średnią długość odpowiedzi
   - B: Miary nieprzewidywalności rozkładu słów
   - C: Liczbę parametrów modelu
   - D: Wskaźnik dokładności klasyfikacji
6. Podaj przykład reguły moderacji treści, którą zastosujesz w systemie.
7. (A–D) Kiedy rozważyć fine-tuning zamiast samego prompt engineering?
   - A: Gdy potrzebna jest adaptacja do specyficznej domeny/stylu
   - B: Zawsze, bo jest „mocniejszy”
   - C: Gdy nie mamy danych
   - D: Gdy chcemy skrócić kontekst
8. Wymień dwie metryki do oceny jakości tekstu generowanego (poza perpleksją).
9. Opisz, jak wymusić format JSON w odpowiedzi modelu i jak go zweryfikować.
10. Wymień trzy biznesowe zastosowania LLM.

## Odpowiedzi (klucz)
1. B
2. Np. brak wiedzy w kontekście, zbyt ogólny prompt, brak walidacji faktów.
3. C
4. Gdy chcemy zademonstrować wzorzec rozwiązania na kilku przykładach dla trudnego zadania.
5. B
6. Np. „Jeśli treść zawiera dane wrażliwe (PESEL, numery kart) — flaguj i nie wyświetlaj”.
7. A
8. Np. BLEU, ROUGE, ocena ekspercka (human evaluation).
9. Opisać schemat pól w promptach, poprosić o wyłącznie JSON, sprawdzić parserem; odrzucić, jeśli nieparsowalny.
10. Chatboty wsparcia, generowanie raportów/dokumentacji, personalizacja treści.