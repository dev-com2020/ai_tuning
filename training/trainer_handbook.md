# Podręcznik trenera — LLM w praktyce

## Cel podręcznika
- Ułatwić prowadzenie szkolenia: przygotowanie, realizację, reagowanie na ryzyka, ewaluację.
- Zawiera checklisty, skrypty mówione, playbooki awaryjne i materiały wzorcowe.

---

## Harmonogram przygotowań (T-minus)
- T−14 dni:
  - Ustal cele z interesariuszami, profil uczestników, wymagania bezpieczeństwa danych.
  - Potwierdź dostęp do kont (OpenAI/HF) i zasady użycia API w firmie.
  - Wyślij wstępną informację o szkoleniu (tematy, wymagania wstępne, sprzęt).
- T−7 dni:
  - Wyślij prework (instalacje, konta, krótkie zadanie diagnostyczne).
  - Zweryfikuj salę/AV: projektor, dźwięk, Wi‑Fi, gniazda zasilania, tablica.
  - Przygotuj dane do labów (zanonimizowane), wygeneruj próbki „złotych odpowiedzi”.
- T−3 dni:
  - Sprawdź środowisko labowe na przykładowej maszynie (czyste środowisko).
  - Wydrukuj ściągi, ćwiczenia, checklisty, formularze ewaluacji.
- T−1 dzień:
  - Zapakuj adaptery, przedłużacze, markery, zapasowy laptop, hotspot.
  - Wyślij przypomnienie z agendą i godziną startu.

---

## Lista kontrolna dnia szkolenia
- Przed startem (−45 do −10 min):
  - Sprawdź AV, Wi‑Fi, dostęp do repo i materiałów.
  - Ustaw układ sali: stoły do pracy w parach, widoczność ekranu, flipchart.
  - Przygotuj login do demo‑kont, otwórz potrzebne karty/przykłady.
- Otwarcie (0–10 min):
  - Przedstaw cele, zasady pracy, sposób zadawania pytań, „parking lot”.
  - Zbierz oczekiwania (sticky notes lub szybka ankieta Mentimeter).
- W trakcie:
  - Timeboxuj segmenty, pilnuj przerw, sygnalizuj kolejne kroki.
  - Monitoruj energię grupy; stosuj mikrozadania i żywe przykłady.
- Zamknięcie dnia:
  - „Exit ticket”: 1 rzecz, którą wynieśli, 1 pytanie na jutro.
  - Przypomnienie o zadaniu domowym (opcjonalnie) i godzinie startu.

---

## Ustawienia sali i sprzętu
- Układ: U‑shape lub klasowy z miejscem na współpracę w parach.
- Sprzęt: projektor 1080p+, mikrofon w większych salach, zapasowe kable/adaptery.
- Tablica/flipchart: do mapowania pomysłów i podsumowań modułów.
- Dostęp do prądu i Wi‑Fi o stabilnym łączu; hotspot jako plan B.

---

## Materiały drukowane i cyfrowe
- Do wydruku: `training/cheatsheet.md`, fragmenty `training/exercises.md`, formularze feedbacku (patrz Załączniki).
- Do projekcji: `training/slides.md` (lub wersja slajdów), przykłady promptów i wyników.
- Do udostępnienia: link do repo, dane przykładowe, referencje.

---

## Konfiguracja środowiska technicznego
- Wymagane narzędzia (preferowane, dostosuj do kontekstu):
  - System: aktualny Linux/macOS/Windows z prawami instalacji.
  - Edytor: VS Code / Cursor; terminal bash; Git.
  - Języki: Python 3.10+ lub Node 18+ (wystarczy jedno środowisko).
  - Biblioteki: klient API LLM (OpenAI/HF), narzędzia JSON (`jq`/`jsonlint`).
- Zmienne środowiskowe (przykład `.env`):
```env
OPENAI_API_KEY=...
HF_TOKEN=...
HTTP_PROXY=
HTTPS_PROXY=
```
- Szybki check:
```bash
python -V || node -v
pip list | head || npm -v
curl -I https://api.openai.com || true
```
- Minimalny workflow (Python):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install openai
python - << 'PY'
print('Env OK')
PY
```

---

## Prowadzenie modułów — wskazówki facylitacyjne
- Otwarcie: przedstaw cele modułu i kryteria sukcesu; pokaż przykład końcowy.
- Techniki aktywizujące: think‑pair‑share, 1‑2‑4‑All, cold/warm call, głosowania.
- Zarządzanie pytaniami: „parking lot”, priorytetyzacja, limit czasu na wątki poboczne.
- Włączanie uczestników cichych: rundka po sali, praca w parach, czat.
- Dyscyplina czasu: zapowiadaj timeboxy, ostrzegaj na 5 min przed końcem.

---

## Zarządzanie energią i inkluzywność
- Mikrozadania co 20–30 min; przerwy co 60–90 min.
- Różnorodne formaty: mini‑demo, ćwiczenie, dyskusja, quiz.
- Zadbaj o dostępność: wielkość czcionek, kontrast slajdów, mikrofon.
- Ustal zasady bezpiecznej dyskusji; reaguj na wykluczające zachowania.

---

## Playbook ryzyk i plan B
- Brak Wi‑Fi:
  - Przejdź do części teorii i ćwiczeń „na sucho”, pokaż nagrane demo.
  - Zaproponuj pracę na lokalnych, wcześniej przygotowanych wynikach.
- Niedostępne API / limity:
  - Przełącz na innego dostawcę lub mniejszy model; ogranicz równoległość do 1.
  - Użyj pre‑zapisanych odpowiedzi do analizy porównawczej.
- Problemy z laptopami:
  - Paruj uczestników, zapewnij zapasowy sprzęt; pracuj w trybie obserwacji.
- Dane wrażliwe:
  - Używaj danych syntetycznych; przypominaj politykę bezpieczeństwa i RODO.

---

## Troubleshooting — ściąga
- Błędy środowiska Python:
  - „Module not found”: aktywuj `.venv`, `pip install -r requirements.txt`.
  - Konflikt wersji: utwórz nowe wirtualne środowisko, zablokuj wersje.
- Błędy API:
  - 401/403: klucz w `.env`, poprawne przekazanie do klienta.
  - 429: dodaj retry/backoff; zmniejsz równoległość.
  - Timeout: sprawdź proxy/firewall, test `curl`.
- JSON nieparsowalny:
  - Wymuś „tylko JSON” i waliduj `jq`/`jsonlint`; poproś o poprawkę.

---

## Skrypty mówione (przykłady)
- Otwarcie szkolenia (2 min):
  - „Cześć! Dziś skupimy się na praktycznym użyciu LLM: jak pisać skuteczne prompty, kontrolować jakość odpowiedzi i kiedy sięgać po fine‑tuning. Na koniec będziecie potrafili… [cele]. Pytania zbieramy na bieżąco, trudniejsze notujemy na parkingu.”
- Przejście do ćwiczeń:
  - „Teraz 15 minut pracy w parach. Najpierw samodzielnie, potem porównajcie wersje i zdecydujcie, co poprawić.”
- Zamknięcie modułu:
  - „Kluczowe punkty: [3 bullet‑y]. Zapiszcie jedno zastosowanie w waszym procesie.”

---

## Ewaluacja i mierzenie postępów
- Szybkie ankiety (Mentimeter/Forms) po modułach.
- „Exit ticket” na koniec dnia (1 wniosek, 1 pytanie).
- Wykorzystaj `training/quiz.md` — przeprowadź quiz, omów odpowiedzi.
- Mini‑projekt: oceniaj według rubryki (kryteria: jakość promptu, kontrola, ewaluacja, uzasadnienie decyzji).

---

## Po szkoleniu
- E‑mail podsumowujący (48 h): materiały, linki, zadania dalsze, kontakt.
- Zaproszenie do kanału społeczności (Slack/Teams) na pytania i wymianę praktyk.
- Propozycja planu 30‑60‑90 (wdrożenia praktyk w zespole).

---

## Załączniki — szablony i checklisty

### A. Checklista przed szkoleniem (skrót)
- Cele, profil uczestników, polityka danych, konta API, sala/AV, materiały.

### B. Checklista startowa
- AV/Internet, widoczność slajdów, miejsca pracy w parach, wydruki, markery.

### C. Checklista przed labem fine‑tuning
- Dostęp do danych, `.env`, biblioteki zainstalowane, próbki „złotych odpowiedzi”.

### D. E‑mail do uczestników (prework — wzór)
- Temat: „Przygotowanie do szkolenia LLM — ważne kroki”
- Treść (skrót): cele, agenda, instalacje (Python/Node, edytor), konta (OpenAI/HF), test połączenia, kontakt w razie problemów.

### E. E‑mail po szkoleniu (follow‑up — wzór)
- Linki do materiałów (`training/*`), repo z przykładami, propozycje dalszych kroków.

### F. Formularz feedbacku (pytania przykładowe)
- Co było najbardziej wartościowe? Co usprawnić? Tempo? Poziom trudności? Czy polecił(a)byś szkolenie i dlaczego?