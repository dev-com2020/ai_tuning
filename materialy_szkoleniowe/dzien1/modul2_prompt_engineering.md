# Moduł 2: Techniki tworzenia efektywnych promptów

## Cel modułu
Po zakończeniu tego modułu uczestnik będzie:
- Znał zasady tworzenia efektywnych promptów
- Stosował zaawansowane techniki promptowania
- Rozpoznawał i unikał typowych błędów
- Optymalizował prompty dla konkretnych zastosowań

## 1. Podstawy Prompt Engineering

### 1.1 Czym jest prompt engineering?

**Definicja**: Sztuka i nauka formułowania instrukcji dla modeli językowych w sposób maksymalizujący jakość i użyteczność odpowiedzi.

**Dlaczego to ważne?**
- Ten sam model może dać diametralnie różne odpowiedzi
- Dobrze sformułowany prompt = 80% sukcesu
- Oszczędność czasu i kosztów (mniej iteracji)
- Większa kontrola nad outputem

### 1.2 Anatomia dobrego promptu

```
[Kontekst/Rola] + [Instrukcja] + [Format] + [Przykłady] + [Ograniczenia]
```

**Przykład rozbicia**:
```
[Rola] Jesteś ekspertem ds. marketingu cyfrowego.
[Instrukcja] Przeanalizuj poniższy post na LinkedIn i zaproponuj 3 ulepszenia.
[Format] Odpowiedź przedstaw w formie listy punktowanej.
[Przykład] Np. "Dodaj call-to-action na końcu"
[Ograniczenia] Każda sugestia max 2 zdania.
```

## 2. Struktura promptu

### 2.1 System Prompt vs User Prompt

**System Prompt** (instrukcja systemowa):
- Definiuje ogólne zachowanie modelu
- Ustala "osobowość" i role
- Pozostaje stały podczas konwersacji

```python
system_prompt = """
Jesteś pomocnym asystentem AI o imieniu TechBot.
Zawsze odpowiadasz w języku polskim.
Jesteś dokładny, rzeczowy i przyjazny.
Unikasz żargonu technicznego, chyba że użytkownik go używa.
"""
```

**User Prompt** (zapytanie użytkownika):
- Konkretne pytanie lub zadanie
- Zmienia się w każdej interakcji
- Może zawierać dodatkowy kontekst

```python
user_prompt = """
Wyjaśnij mi różnicę między RAM a dyskiem twardym.
Moja babcia nie zna się na komputerach.
"""
```

### 2.2 Komponenty efektywnego promptu

1. **Jasny cel**
   ```
   ❌ "Napisz coś o AI"
   ✅ "Napisz 300-słowny artykuł o zastosowaniach AI w medycynie"
   ```

2. **Konkretny kontekst**
   ```
   ❌ "Jak zwiększyć sprzedaż?"
   ✅ "Jak zwiększyć sprzedaż w sklepie internetowym z odzieżą sportową, skierowanym do kobiet 25-40 lat?"
   ```

3. **Określony format**
   ```
   ❌ "Daj mi pomysły"
   ✅ "Wymień 5 pomysłów w formie krótkiej listy punktowanej"
   ```

4. **Przykłady (few-shot)**
   ```
   "Przekształć zdania na bardziej formalne:
   Przykład: 'To jest super' → 'To jest doskonałe'
   Teraz przekształć: 'Ten pomysł jest spoko'"
   ```

## 3. Zaawansowane techniki promptowania

### 3.1 Chain-of-Thought (CoT) Prompting

**Zasada**: Prowadź model przez proces myślowy krok po kroku.

**Przykład standardowy**:
```
Prompt: "W sklepie było 23 jabłka. Sprzedano 17. Ile zostało?"
Odpowiedź: "6"
```

**Przykład z CoT**:
```
Prompt: "W sklepie było 23 jabłka. Sprzedano 17. Ile zostało? 
Rozwiąż krok po kroku."

Odpowiedź: 
"Krok 1: Zacznijmy od liczby jabłek w sklepie: 23
Krok 2: Liczba sprzedanych jabłek: 17
Krok 3: Aby znaleźć ile zostało, odejmujemy: 23 - 17 = 6
Odpowiedź: W sklepie zostało 6 jabłek."
```

### 3.2 Zero-shot vs Few-shot Learning

**Zero-shot** - bez przykładów:
```
"Klasyfikuj poniższy tekst jako pozytywny, neutralny lub negatywny:
'Produkt spełnił moje oczekiwania.'"
```

**One-shot** - jeden przykład:
```
"Klasyfikuj sentyment:
Przykład: 'Jestem zachwycony!' → pozytywny
Klasyfikuj: 'Produkt spełnił moje oczekiwania.'"
```

**Few-shot** - kilka przykładów:
```
"Klasyfikuj sentyment:
'Jestem zachwycony!' → pozytywny
'Produkt jest OK' → neutralny  
'Totalna porażka' → negatywny
'To było przeciętne' → ?"
```

### 3.3 Role-playing (odgrywanie ról)

**Technika**: Przypisz modelowi konkretną rolę lub eksperta.

```
"Jesteś doświadczonym programistą Python z 15-letnim stażem.
Specjalizujesz się w optymalizacji kodu.
Przeanalizuj poniższy kod i zasugeruj optymalizacje."
```

**Przykłady ról**:
- Ekspert dziedzinowy: "Jesteś kardiologiem..."
- Persona: "Jesteś przyjaźnie nastawionym nauczycielem..."
- Fikcyjna postać: "Odpowiedz jak Sherlock Holmes..."

### 3.4 Self-Consistency

**Zasada**: Generuj wiele odpowiedzi i wybierz najczęstszą.

```python
prompts = [
    "Rozwiąż: 127 + 389 = ?",
    "Oblicz sumę 127 i 389",
    "Dodaj 127 do 389"
]
# Zbierz odpowiedzi i wybierz najczęstszą
```

### 3.5 Prompt Chaining

**Zasada**: Łącz wiele promptów w sekwencję.

```
Prompt 1: "Wymień 5 głównych wyzwań w zarządzaniu projektami IT"
↓
Prompt 2: "Dla wyzwania nr 3 z poprzedniej listy, zaproponuj 3 rozwiązania"
↓
Prompt 3: "Stwórz plan wdrożenia dla rozwiązania nr 2"
```

## 4. Formatowanie i struktura

### 4.1 Używanie delimitatorów

```
Przeanalizuj tekst między potrójnymi kreskami:
---
Lorem ipsum dolor sit amet...
---
```

**Popularne delimitatory**:
- ``` ``` - dla kodu
- """ """ - dla długich tekstów
- ### ### - dla sekcji
- --- --- - dla separacji

### 4.2 Strukturyzacja OUTPUT

```
"Przygotuj analizę SWOT dla startup'u technologicznego.
Format odpowiedzi:

## MOCNE STRONY
- [punkt 1]
- [punkt 2]

## SŁABE STRONY
- [punkt 1]
- [punkt 2]

[analogicznie dla Szans i Zagrożeń]"
```

### 4.3 Formaty danych

**JSON**:
```
"Wygeneruj dane 3 użytkowników w formacie JSON:
{
  "id": number,
  "name": string,
  "email": string,
  "age": number
}"
```

**Markdown**:
```
"Sformatuj odpowiedź używając Markdown:
- Nagłówki dla sekcji (##)
- Listy punktowane dla elementów
- **Pogrubienie** dla kluczowych terminów"
```

**CSV**:
```
"Przedstaw dane w formacie CSV:
Nagłówek: Imię,Wiek,Miasto
Wygeneruj 5 przykładowych rekordów"
```

## 5. Przykłady dobrych i złych praktyk

### 5.1 Jasność i precyzja

❌ **Źle**:
```
"Napisz coś o pogodzie"
```
- Brak kontekstu
- Nieokreślony format
- Niejasny cel

✅ **Dobrze**:
```
"Napisz 3-akapitową prognozę pogody dla Warszawy na jutro.
Uwzględnij temperaturę, opady i wiatr.
Ton: profesjonalny ale przystępny."
```

### 5.2 Unikanie dwuznaczności

❌ **Źle**:
```
"Jak naprawić komputer?"
```

✅ **Dobrze**:
```
"Laptop Dell Latitude nie włącza się po naciśnięciu przycisku power.
Bateria jest naładowana, zasilacz podłączony.
Jakie kroki diagnostyczne powinienem wykonać?"
```

### 5.3 Kontrola długości

❌ **Źle**:
```
"Wyjaśnij mi blockchain"
```

✅ **Dobrze**:
```
"Wyjaśnij blockchain w 3 zdaniach.
Użyj prostego języka odpowiedniego dla osoby nietechnicznej.
Skup się na praktycznym zastosowaniu."
```

## 6. Techniki optymalizacji

### 6.1 Iteracyjne doskonalenie

```
Wersja 1: "Napisz email"
↓
Wersja 2: "Napisz email biznesowy"
↓
Wersja 3: "Napisz email do klienta z przeprosinami za opóźnienie"
↓
Wersja 4: "Napisz profesjonalny email do klienta korporacyjnego
z przeprosinami za 3-dniowe opóźnienie dostawy.
Ton: profesjonalny ale empatyczny.
Długość: 100-150 słów.
Zawrzyj propozycję rekompensaty."
```

### 6.2 A/B Testing promptów

**Test A**:
```
"Jesteś ekspertem SEO. Zaproponuj tytuł artykułu o kawie."
```

**Test B**:
```
"Stwórz angażujący tytuł artykułu o kawie.
Cel: wysoka klikalność w Google.
Zawrzyj: główne słowo kluczowe 'najlepsza kawa'.
Długość: 50-60 znaków."
```

### 6.3 Prompt Templates

```python
def create_analysis_prompt(topic, aspects, format="bullet points"):
    return f"""
    Przeprowadź analizę tematu: {topic}
    
    Uwzględnij następujące aspekty:
    {' '.join([f'- {asp}' for asp in aspects])}
    
    Format odpowiedzi: {format}
    Długość: 200-300 słów
    Język: profesjonalny ale przystępny
    """
```

## 7. Przypadki użycia

### 7.1 Content Creation

```
"Rola: Jesteś content writerem specjalizującym się w technologii.
Zadanie: Napisz wprowadzenie do artykułu o sztucznej inteligencji w edukacji.
Wymagania:
- Hook w pierwszym zdaniu
- 3 kluczowe punkty do rozwinięcia
- Pytanie angażujące na końcu
- Długość: 150-200 słów"
```

### 7.2 Data Analysis

```
"Przeanalizuj poniższe dane sprzedażowe:
[dane]

Odpowiedz na pytania:
1. Jaki jest trend sprzedaży?
2. Który produkt sprzedaje się najlepiej?
3. W którym miesiącu była najwyższa sprzedaż?

Przedstaw wnioski w 3 punktach."
```

### 7.3 Code Generation

```
"Napisz funkcję Python która:
- Przyjmuje listę liczb jako argument
- Zwraca słownik z statystykami: min, max, średnia, mediana
- Obsługuje błędy (pusta lista, nie-liczby)
- Dodaj docstring i type hints
- Użyj biblioteki statistics dla mediany"
```

## 8. Ćwiczenia praktyczne

### Ćwiczenie 1: Prompt Evolution
1. Zacznij od prostego promptu: "Napisz o kotach"
2. Iteracyjnie ulepszaj, dodając:
   - Kontekst
   - Format
   - Długość
   - Ton
   - Cel
3. Porównaj wyniki

### Ćwiczenie 2: Role-playing Challenge
1. Wybierz temat (np. "inwestowanie")
2. Stwórz 3 prompty z różnymi rolami:
   - Konserwatywny doradca finansowy
   - Agresywny trader
   - Nauczyciel ekonomii
3. Porównaj różnice w odpowiedziach

### Ćwiczenie 3: Few-shot Learning
1. Zadanie: Klasyfikacja emaili (spam/nie-spam)
2. Stwórz prompt z:
   - 0 przykładów
   - 1 przykładem
   - 3 przykładami
   - 5 przykładami
3. Testuj na 10 emailach, porównaj dokładność

### Ćwiczenie 4: Chain-of-Thought
1. Problem: "W firmie pracuje 120 osób. 40% to kobiety. 
   25% kobiet i 30% mężczyzn ma wyższe wykształcenie. 
   Ile osób ma wyższe wykształcenie?"
2. Napisz prompt bez CoT
3. Napisz prompt z CoT
4. Porównaj dokładność i klarowność

## 9. Narzędzia i zasoby

### 9.1 Prompt Libraries
- **PromptBase** - marketplace promptów
- **Awesome Prompts** - GitHub repository
- **ShareGPT** - społeczność dzieląca się promptami

### 9.2 Testing Tools
```python
# Prosty framework do testowania promptów
class PromptTester:
    def __init__(self, model):
        self.model = model
        
    def test_variations(self, base_prompt, variations):
        results = []
        for variation in variations:
            response = self.model.generate(base_prompt + variation)
            results.append({
                'variation': variation,
                'response': response,
                'quality_score': self.evaluate_quality(response)
            })
        return results
```

### 9.3 Prompt Management
```yaml
# prompt_library.yaml
customer_service:
  greeting:
    system: "Jesteś przyjaznym konsultantem..."
    temperature: 0.7
  complaint_handling:
    system: "Jesteś empatycznym specjalistą..."
    temperature: 0.3
```

## 10. Typowe pułapki i jak ich unikać

### 10.1 Overloading
❌ Próba załatwienia wszystkiego w jednym prompcie
✅ Rozbij na mniejsze, focused prompts

### 10.2 Assumption Making
❌ Zakładanie, że model "wie" o czym mówisz
✅ Zawsze dostarczaj kontekst

### 10.3 Ignoring Model Limitations
❌ Oczekiwanie perfekcji w bardzo specjalistycznych dziedzinach
✅ Weryfikuj fakty, szczególnie liczby i daty

### 10.4 Inconsistent Formatting
❌ Różne style w jednym prompcie
✅ Zachowaj spójność w formatowaniu i stylu

## 11. Metryki skuteczności

### 11.1 Relevance Score
- Czy odpowiedź jest na temat?
- Skala 1-5

### 11.2 Completeness
- Czy wszystkie punkty zostały zaadresowane?
- Checklist approach

### 11.3 Accuracy
- Faktyczna poprawność
- Szczególnie ważne dla danych/liczb

### 11.4 Usability
- Czy odpowiedź jest praktycznie użyteczna?
- Czy wymaga dodatkowej edycji?

## 12. Zadanie końcowe

Stwórz kompletny system promptów dla chatbota obsługi klienta:
1. System prompt definiujący osobowość
2. Prompty dla 5 typowych scenariuszy:
   - Powitanie
   - Zapytanie o produkt
   - Reklamacja
   - Pomoc techniczna
   - Pożegnanie
3. Instrukcje eskalacji do człowieka
4. Test na 3 przykładowych konwersacjach

## Podsumowanie

1. **Prompt engineering to iteracyjny proces**
2. **Kontekst i jasność są kluczowe**
3. **Różne techniki dla różnych zadań**
4. **Testowanie i optymalizacja są niezbędne**
5. **Jedna technika nie rozwiązuje wszystkich problemów**

## Materiały do dalszej nauki

- "The Prompt Engineering Guide" - Dair AI
- "Best Practices for Prompt Engineering" - OpenAI
- "Prompt Engineering Course" - DeepLearning.AI
- "Chain-of-Thought Prompting Papers" - Google Research