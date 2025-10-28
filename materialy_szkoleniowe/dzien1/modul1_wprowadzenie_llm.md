# Moduł 1: Zrozumienie dużych modeli językowych

## Cel modułu
Po zakończeniu tego modułu uczestnik będzie:
- Rozumiał podstawowe koncepcje związane z dużymi modelami językowymi
- Znał główne architektury i ich zastosowania
- Identyfikował kluczowe problemy i ograniczenia LLM

## 1. Wprowadzenie do LLM

### 1.1 Czym są duże modele językowe?
Duże modele językowe (Large Language Models - LLM) to zaawansowane systemy sztucznej inteligencji trenowane na ogromnych zbiorach tekstowych, zdolne do:
- Rozumienia i generowania tekstu w języku naturalnym
- Odpowiadania na pytania
- Tłumaczenia między językami
- Pisania kodu
- Analizy i streszczania dokumentów

### 1.2 Historia i ewolucja
- **2017**: Wprowadzenie architektury Transformer (Vaswani et al.)
- **2018**: BERT - rewolucja w rozumieniu języka
- **2019**: GPT-2 - kontrowersje wokół generowania tekstu
- **2020**: GPT-3 - skok jakościowy w możliwościach
- **2022**: ChatGPT - demokratyzacja dostępu do LLM
- **2023-2024**: Claude, GPT-4, Gemini - wyścig technologiczny

## 2. Architektury dużych modeli językowych

### 2.1 Transformer - fundament współczesnych LLM

#### Kluczowe komponenty:
1. **Self-Attention Mechanism**
   - Pozwala modelowi "patrzeć" na wszystkie słowa jednocześnie
   - Ustala relacje między słowami w zdaniu
   - Multi-head attention dla różnych perspektyw

2. **Positional Encoding**
   - Dodaje informację o pozycji słów
   - Kluczowe dla zrozumienia sekwencji

3. **Feed-Forward Networks**
   - Przetwarzanie informacji z attention
   - Nieliniowe transformacje

#### Przykład działania:
```
Wejście: "Kot siedzi na macie"
Tokenizacja: ["Kot", "siedzi", "na", "macie"]
Attention: Model ustala że "siedzi" odnosi się do "Kot", "na" łączy "siedzi" z "macie"
```

### 2.2 GPT (Generative Pre-trained Transformer)

#### Charakterystyka:
- **Architektura**: Decoder-only Transformer
- **Trening**: Autoregresywny (przewidywanie następnego tokenu)
- **Mocne strony**: Generowanie tekstu, kreatywność
- **Wersje**: GPT-3 (175B parametrów), GPT-4 (nieujawniona liczba)

#### Zastosowania:
- Pisanie artykułów i esejów
- Generowanie kodu
- Chatboty konwersacyjne
- Pomoc w kreatywnym pisaniu

### 2.3 Gemini (Google)

#### Charakterystyka:
- **Multimodalność**: Tekst, obrazy, audio, wideo
- **Różne rozmiary**: Nano, Pro, Ultra
- **Integracja**: Ścisła integracja z ekosystemem Google

#### Zastosowania:
- Analiza dokumentów multimedialnych
- Asystent w Google Workspace
- Zaawansowane wyszukiwanie

### 2.4 Claude (Anthropic)

#### Charakterystyka:
- **Constitutional AI**: Wbudowane zasady etyczne
- **Długi kontekst**: Do 200k tokenów
- **Bezpieczeństwo**: Priorytet na helpful, harmless, honest

#### Zastosowania:
- Analiza długich dokumentów
- Zadania wymagające etycznego podejścia
- Współpraca w projektach badawczych

## 3. Zasada działania LLM

### 3.1 Proces treningu

1. **Pre-training**
   ```
   Dane wejściowe: Miliardy stron tekstu z internetu
   Cel: Nauczyć model przewidywać następne słowo
   Metoda: Unsupervised learning
   ```

2. **Fine-tuning**
   ```
   Dane wejściowe: Wyselekcjonowane przykłady z etykietami
   Cel: Dostosować model do konkretnych zadań
   Metoda: Supervised learning
   ```

3. **RLHF (Reinforcement Learning from Human Feedback)**
   ```
   Dane wejściowe: Oceny ludzkie odpowiedzi
   Cel: Dopasować model do ludzkich preferencji
   Metoda: Reinforcement learning
   ```

### 3.2 Tokenizacja

Przykład tokenizacji:
```
Tekst: "Sztuczna inteligencja"
Tokeny: ["Szt", "ucz", "na", " ", "int", "eli", "gen", "cja"]
Token IDs: [1234, 5678, 910, 11, 1213, 1415, 1617, 1819]
```

### 3.3 Generowanie odpowiedzi

1. **Sampling strategies**:
   - **Greedy**: Zawsze wybiera najbardziej prawdopodobne słowo
   - **Top-k**: Wybiera z k najbardziej prawdopodobnych
   - **Top-p (nucleus)**: Wybiera z tokenów sumujących się do p prawdopodobieństwa
   - **Temperature**: Kontroluje "kreatywność" (0 = deterministyczne, 1+ = kreatywne)

## 4. Kluczowe problemy

### 4.1 Halucynacje

**Definicja**: Model generuje informacje, które brzmią przekonująco, ale są nieprawdziwe.

**Przykłady**:
- Zmyślone cytaty i źródła
- Nieistniejące fakty historyczne
- Błędne obliczenia matematyczne

**Przyczyny**:
- Brak rzeczywistego "rozumienia"
- Statystyczne dopasowanie wzorców
- Nadmierna pewność modelu

**Metody mitygacji**:
1. Weryfikacja faktów
2. Prośba o źródła
3. Cross-checking z innymi źródłami
4. Używanie temperature = 0 dla faktów

### 4.2 Tendencyjność (Bias)

**Typy bias**:
- **Kulturowy**: Odzwierciedlenie zachodnich wartości
- **Językowy**: Lepsza wydajność w angielskim
- **Demograficzny**: Stereotypy dotyczące płci, rasy
- **Temporalny**: Przestarzałe informacje

**Przykład wykrywania bias**:
```
Prompt: "Opisz typowego programistę"
Problematyczna odpowiedź: "Typowy programista to młody mężczyzna..."
Lepsza odpowiedź: "Programiści to różnorodna grupa ludzi..."
```

### 4.3 Ograniczenia kontekstu

**Problem**: Modele mają ograniczoną "pamięć" (context window)
- GPT-3.5: 4,096 tokenów
- GPT-4: 8,192 - 128,000 tokenów
- Claude: 100,000 - 200,000 tokenów

**Konsekwencje**:
- Utrata informacji w długich konwersacjach
- Niemożność analizy bardzo długich dokumentów
- Konieczność summaryzacji

## 5. Demonstracja praktyczna

### 5.1 Porównanie modeli

Testowy prompt:
```
"Wyjaśnij czym jest kwantowa superpozycja w sposób zrozumiały dla 10-latka"
```

**GPT-4 Response**:
"Wyobraź sobie, że masz magiczną monetę..."

**Claude Response**:
"Kwantowa superpozycja to jak..."

**Gemini Response**:
"To trochę jak gdy..."

### 5.2 Identyfikacja halucynacji

Ćwiczenie:
1. Zadaj modelowi pytanie o nieistniejące wydarzenie
2. Obserwuj czy model przyzna się do niewiedzy
3. Analizuj sygnały ostrzegawcze

## 6. Ćwiczenia praktyczne

### Ćwiczenie 1: Eksploracja modeli
1. Wybierz 3 różne modele (GPT, Claude, Gemini)
2. Zadaj im to samo złożone pytanie
3. Porównaj:
   - Styl odpowiedzi
   - Dokładność
   - Długość
   - Ton

### Ćwiczenie 2: Wykrywanie halucynacji
1. Poproś model o:
   - Cytat z nieistniejącej książki
   - Opis nieistniejącego wydarzenia historycznego
   - Rozwiązanie niemożliwego problemu matematycznego
2. Dokumentuj odpowiedzi
3. Analizuj techniki unikania halucynacji

### Ćwiczenie 3: Testowanie bias
1. Przygotuj serię promptów testujących bias:
   - Zawodowy
   - Kulturowy
   - Płciowy
2. Analizuj odpowiedzi
3. Zaproponuj lepsze sformułowania

## 7. Kluczowe wnioski

1. **LLM to potężne narzędzia, ale nie są nieomylne**
2. **Zrozumienie architektury pomaga w efektywnym wykorzystaniu**
3. **Świadomość ograniczeń jest kluczowa dla bezpiecznego użycia**
4. **Różne modele mają różne mocne strony**

## 8. Pytania do dyskusji

1. Jakie widzisz największe zagrożenia związane z LLM?
2. W jaki sposób architektura wpływa na możliwości modelu?
3. Jak możemy wykorzystać różnice między modelami?
4. Czy halucynacje to zawsze problem, czy czasem feature?

## 9. Zadanie domowe

1. Przeprowadź test Turinga z wybranym modelem
2. Znajdź 3 przykłady halucynacji w odpowiedziach LLM
3. Porównaj odpowiedzi 2 modeli na pytanie z twojej dziedziny
4. Przygotuj listę pytań na następne zajęcia

## 10. Materiały dodatkowe

### Artykuły naukowe:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "Constitutional AI" (Anthropic, 2022)

### Blogi i tutoriale:
- "The Illustrated Transformer" - Jay Alammar
- "How GPT Works" - OpenAI Blog
- "Understanding LLM Hallucinations" - Anthropic Research

### Narzędzia online:
- [OpenAI Playground](https://platform.openai.com/playground)
- [Claude.ai](https://claude.ai)
- [Google AI Studio](https://makersuite.google.com)