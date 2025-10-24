# Instrukcja dla Trenera - Szkolenie NLP

## 📋 Przygotowanie przed szkoleniem

### Tydzień przed szkoleniem

- [ ] Wyślij uczestnikom email z instrukcjami instalacji
- [ ] Poproś o instalację Python 3.8+ i Jupyter
- [ ] Przekaż listę wymaganych bibliotek (requirements.txt)
- [ ] Poproś o zainstalowanie modeli spaCy
- [ ] Zbierz informacje o poziomie uczestników
- [ ] Przygotuj przykłady z branży uczestników (jeśli możliwe)

### Dzień przed szkoleniem

- [ ] Sprawdź czy wszystkie notebooki działają
- [ ] Przetestuj wszystkie przykłady kodu
- [ ] Przygotuj dodatkowe zadania dla szybszych uczestników
- [ ] Sprawdź dostęp do internetu (potrzebny do pobierania modeli)
- [ ] Przygotuj backup modeli offline (na wypadek problemów z siecią)

### Materiały dodatkowe

- [ ] Przygotuj slajdy do wprowadzenia (opcjonalne)
- [ ] Przykłady z realnych projektów
- [ ] Case studies z branży
- [ ] Lista dodatkowych zasobów

---

## ⏰ Harmonogram szkolenia

### Dzień 1: Podstawy NLP

#### 9:00 - 9:30: Wprowadzenie (30 min)
- Przedstawienie się
- Omówienie programu
- Oczekiwania uczestników
- Sprawdzenie środowisk

#### 9:30 - 10:45: Moduł 1 - Wprowadzenie do NLP (75 min)
**Kluczowe punkty:**
- Definicja NLP i wyzwania
- Przegląd zastosowań (pokazać realne przykłady!)
- Ewolucja: od reguł do Transformerów
- Prosty przykład sentiment analysis

**Tips:**
- Zachęć do dyskusji o zastosowaniach w ich pracy
- Pokaż wideo: Google Translate in action
- Demo: ChatGPT jako przykład nowoczesnego NLP

#### 10:45 - 11:00: Przerwa ☕

#### 11:00 - 12:30: Moduł 2 - Narzędzia i biblioteki (90 min)
**Kluczowe punkty:**
- NLTK - demo podstawowych funkcji
- spaCy - pipeline i szybkość
- Hugging Face - Model Hub
- OpenAI API - potęga i łatwość

**Tips:**
- Porównaj na żywo szybkość NLTK vs spaCy
- Pokaż Hugging Face Model Hub w przeglądarce
- Jeśli masz API key, pokaż live ChatGPT API call
- Podkreśl kiedy używać której biblioteki

#### 12:30 - 13:30: Przerwa obiadowa 🍽️

#### 13:30 - 15:00: Moduł 3 - Podstawowe operacje (90 min)
**Kluczowe punkty:**
- Tokenizacja - dlaczego to nie jest proste!
- Stemming vs Lematyzacja - kiedy co?
- Stop words - kiedy usuwać, kiedy nie
- POS tagging i dependency parsing

**Tips:**
- Pokaż edge cases tokenizacji (emoji, URL)
- Demo: wizualizacja dependency tree w spaCy
- Niech uczestnicy spróbują na własnych tekstach
- Omów błędy i trudne przypadki

#### 15:00 - 15:15: Przerwa ☕

#### 15:15 - 17:00: Warsztaty praktyczne (105 min)
**Projekty:**
1. Analiza sentymentu recenzji (30 min)
2. Spam detection (30 min)
3. Mini-projekt grupowy (45 min)

**Tips:**
- Podziel uczestników na grupy 2-3 osobowe
- Krąż i pomagaj przy problemach
- Zachęć do eksperymentowania
- Na koniec: prezentacja 1-2 projektów

#### 17:00 - 17:15: Podsumowanie Dnia 1
- Recap najważniejszych punktów
- Q&A
- Zadanie domowe (opcjonalne): eksperymentuj z własnymi danymi

---

### Dzień 2: Zaawansowane modele i biznes

#### 9:00 - 9:15: Powitanie i recap Dnia 1 (15 min)
- Krótkie przypomnienie
- Odpowiedzi na pytania
- Plan na dziś

#### 9:15 - 10:45: Moduł 4 - Transformery i modele (90 min)
**Kluczowe punkty:**
- Architektura Transformer - wyjaśnij attention
- BERT - rozumienie kontekstu
- GPT - generowanie tekstu
- T5 - uniwersalność text-to-text
- Fine-tuning - kiedy i jak

**Tips:**
- Wizualizuj attention matrix (użyj notebooka!)
- Pokaż różnicę: BERT fill-mask vs GPT generation
- Demo: porównaj różne rozmiary modeli
- Wyjaśnij transfer learning na prostym przykładzie
- Fine-tuning: pokaż kod, ale nie trenuj na żywo (za długo)

#### 10:45 - 11:00: Przerwa ☕

#### 11:00 - 12:30: Moduł 5 - Generowanie i rozumienie (90 min)
**Kluczowe punkty:**
- Summarization: extractive vs abstractive
- Text generation: temperature i inne parametry
- Translation: rozwój od SMT do Transformers
- Sentiment analysis: różne poziomy
- Question Answering: budowa systemu Q&A

**Tips:**
- Pokaż na żywo wpływ temperature na generowanie
- Demo: przetłumacz ten sam tekst na kilka języków
- Aspect-based sentiment - przykład z restauracji
- Q&A: zbuduj prostą FAQ bot

#### 12:30 - 13:30: Przerwa obiadowa 🍽️

#### 13:30 - 15:30: Moduł 6 - NLP w biznesie (120 min)
**Kluczowe punkty:**
- Chatboty: rule-based → ML-based → generative
- Automatyzacja dokumentów: faktury, CV, kontrakty
- Personalizacja komunikacji
- Case study: kompletny system support

**Tips:**
- Pokaż różnicę między typami chatbotów
- Demo: ekstrakcja z faktury krok po kroku
- Omów etykę i bias w personalizacji
- System support: przeprowadź przez cały flow
- Podkreśl best practices dla produkcji

#### 15:30 - 15:45: Przerwa ☕

#### 15:45 - 16:45: Projekt końcowy (60 min)
**Opcje:**
1. Budowa chatbota dla konkretnej branży
2. System analizy dokumentów
3. Pipeline personalizacji
4. Własny pomysł uczestników

**Tips:**
- Niech wybiorą projekt bliski ich potrzebom biznesowym
- Praca w grupach 2-3 osobowych
- Dostępny do pomocy
- Cel: działający prototyp

#### 16:45 - 17:15: Prezentacje i podsumowanie (30 min)
- Krótkie prezentacje projektów (5 min/grupa)
- Feedback
- Omówienie dalszych kroków
- Q&A
- Rozdanie certyfikatów

---

## 🎯 Wskazówki dydaktyczne

### Dla różnych poziomów uczestników

**Początkujący:**
- Więcej czasu na podstawy
- Skupić się na gotowych rozwiązaniach (pipelines)
- Mniej teorii, więcej praktyki
- Pomijaj zaawansowane szczegóły architektury

**Średniozaawansowani:**
- Balansuj teorię i praktykę
- Omów szczegóły implementacji
- Zachęć do eksperymentowania z parametrami
- Dyskutuj o trade-offach

**Zaawansowani:**
- Więcej o architekturze modeli
- Szczegóły fine-tuningu
- Optymalizacja i deployment
- Najnowsze research papers

### Techniki angażowania

1. **Live coding**: Pisz kod na żywo, nie tylko pokazuj
2. **Błędy**: Celowo popełnij błąd i napraw go
3. **Pytania**: Zadawaj pytania grupie przed pokazaniem odpowiedzi
4. **Real examples**: Używaj przykładów z ich branży
5. **Humor**: NLP ma swoje zabawne momenty (błędy tokenizacji, translation fails)

### Zarządzanie czasem

- ⏰ Używaj timera dla ćwiczeń
- 📊 Miej backup slides na wypadek skrócenia
- 🚀 Przygotuj dodatkowe zadania dla szybszych
- ⏸️ Elastycznie dostosowuj tempo do grupy

---

## 🔧 Troubleshooting podczas szkolenia

### Częste problemy i rozwiązania

**Problem: Ktoś nie ma zainstalowanych bibliotek**
- Miej przygotowane środowisko Google Colab jako backup
- Lub: share swojego screena i niech śledzi

**Problem: Model nie ściąga się**
- Miej modele ściągnięte lokalnie
- Użyj mniejszych modeli jako alternatywy
- Google Colab ma większość modeli już ściągniętych

**Problem: Kod działa za wolno**
- Zmniejsz batch_size
- Użyj mniejszych modeli (distilbert zamiast bert)
- Ogranicz długość tekstów

**Problem: Out of Memory**
- Restartuj kernel
- Użyj mniejszych modeli
- Przetwarzaj w mniejszych batch'ach

**Problem: Różnice Windows/Mac/Linux**
- Testuj kod na wszystkich systemach przed szkoleniem
- Używaj cross-platform path handling

---

## 📊 Ocena efektywności szkolenia

### Po każdym module (szybka ocena)
- "Czy to było jasne?" (kciuki w górę/dół)
- "Pytania?"
- Obserwuj body language

### Koniec dnia
- Krótka ankieta (3 pytania)
- Co było najciekawsze?
- Co było niejasne?
- Czego chcą więcej jutro?

### Koniec szkolenia
- Pełna ankieta ewaluacyjna
- Oceń: treść, tempo, przykłady, prowadzenie
- Zbierz sugestie na przyszłość

---

## 📚 Dodatkowe materiały do rozdania

### Checklista dla uczestników
- [ ] Zainstalowane środowisko
- [ ] Ściągnięte modele spaCy
- [ ] Przeczytane README
- [ ] Wykonane wszystkie ćwiczenia z Dnia 1
- [ ] Wykonane wszystkie ćwiczenia z Dnia 2
- [ ] Zbudowany własny mini-projekt
- [ ] Zapisane notatki
- [ ] Dodane zakładki do zasobów

### Roadmap dalszej nauki
1. Tydzień 1-2: Powtórka materiałów, własne eksperymenty
2. Tydzień 3-4: Zbuduj pierwszy projekt produkcyjny
3. Miesiąc 2: Fine-tuning na własnych danych
4. Miesiąc 3+: Zaawansowane tematy (RAG, LLM agents)

---

## 🎓 Certyfikaty

Szablon certyfikatu powinien zawierać:
- Imię i nazwisko uczestnika
- Tytuł szkolenia: "Natural Language Processing - od podstaw do zastosowań biznesowych"
- Data szkolenia
- Liczba godzin (14h)
- Podpis prowadzącego
- Logo firmy szkoleniowej

---

## 📞 Contact & Follow-up

### Po szkoleniu
- Wyślij uczestnikom:
  - Link do materiałów (jeśli online)
  - Dodatkowe zasoby
  - Odpowiedzi na pytania, które pozostały
  - Ankietę ewaluacyjną (jeśli nie wypełnili)

### Follow-up (po 2 tygodniach)
- Email: "Jak idzie z projektami?"
- Zaproś do grupy LinkedIn/Discord
- Zaproponuj sesję Q&A online

---

## ✅ Checklist na dzień szkolenia

### Rano (przed startem)
- [ ] Laptop naładowany + zapasowa ładowarka
- [ ] Adapter HDMI/USB-C do projektora
- [ ] Backup materiałów na USB/cloud
- [ ] Woda/kawa dla siebie
- [ ] Kartki i długopisy dla uczestników
- [ ] Lista obecności
- [ ] Certyfikaty wydrukowane
- [ ] Test rzutnika i dźwięku

### Materiały cyfrowe
- [ ] Wszystkie notebooki przetestowane
- [ ] Modele ściągnięte lokalnie
- [ ] Google Colab backup przygotowany
- [ ] Przykładowe dane w katalogu
- [ ] README zaktualizowane

---

## 💡 Pro Tips

1. **Energia**: Utrzymuj wysoką energię, szczególnie po obiedzie
2. **Przerwy**: Są kluczowe! Nie skracaj ich
3. **Pytania**: "Głupich pytań nie ma" - stwórz bezpieczną atmosferę
4. **Tempo**: Lepiej wolniej i solidnie niż szybko i powierzchownie
5. **Praktyka**: 70% praktyki, 30% teorii
6. **Stories**: Opowiadaj o realnych projektach i problemach
7. **Humor**: Śmiej się z błędów (swoich też!)
8. **Feedback**: Pytaj regularnie czy tempo jest OK
9. **Networking**: Zachęć uczestników do wymiany kontaktów
10. **Followup**: Nie kończ kontaktu po szkoleniu

---

**Powodzenia w prowadzeniu szkolenia! 🎉**

Pamiętaj: Najlepsze szkolenia to te, gdzie uczestnicy są aktywni, zadają pytania i budują własne projekty!
