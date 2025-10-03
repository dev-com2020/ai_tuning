# Moduł 3: Kontrolowanie jakości i bezpieczeństwa generowanych treści

## Cel modułu
Po zakończeniu tego modułu uczestnik będzie:
- Rozumiał i stosował metody ograniczania halucynacji
- Kontrolował styl, ton i zakres generowanych treści  
- Implementował zabezpieczenia i moderację automatyczną
- Projektował bezpieczne systemy oparte na LLM

## 1. Metody ograniczania ryzyka halucynacji

### 1.1 Zrozumienie problemu halucynacji

**Definicja**: Halucynacje to sytuacje, gdy model generuje informacje brzmiące przekonująco, ale nieprawdziwe lub nieistniejące.

**Typy halucynacji**:
1. **Faktyczne** - nieprawdziwe fakty, daty, liczby
2. **Źródłowe** - nieistniejące cytaty, publikacje
3. **Logiczne** - błędne wnioskowanie
4. **Temporalne** - anachronizmy, błędne sekwencje

### 1.2 Parametry kontrolujące generowanie

#### Temperature (Temperatura)
Kontroluje losowość/kreatywność odpowiedzi.

```python
# Temperature = 0 (deterministyczne)
prompt = "Stolica Polski to"
# Odpowiedź: "Warszawa" (zawsze ta sama)

# Temperature = 0.7 (zbalansowane)  
prompt = "Napisz krótką historię o kocie"
# Odpowiedź: Zróżnicowana, ale sensowna

# Temperature = 1.5 (bardzo kreatywne)
prompt = "Wymyśl nowe słowo i jego definicję"
# Odpowiedź: Bardzo kreatywna, może być chaotyczna
```

**Zastosowania Temperature**:
- `0.0 - 0.3`: Fakty, analizy, odpowiedzi techniczne
- `0.4 - 0.7`: Ogólne zastosowania, balans
- `0.8 - 1.0`: Kreatywne pisanie, brainstorming
- `1.0+`: Eksperymentalne, bardzo kreatywne

#### Top-p (Nucleus Sampling)
Wybiera tokeny z górnego percentyla prawdopodobieństwa.

```python
# Top-p = 0.1 (bardzo restrykcyjne)
# Wybiera tylko z top 10% najbardziej prawdopodobnych tokenów

# Top-p = 0.9 (standardowe)
# Wybiera z tokenów sumujących się do 90% prawdopodobieństwa

# Top-p = 1.0 (wszystkie tokeny)
# Brak ograniczeń
```

#### Frequency Penalty
Karze za powtarzanie tokenów.

```python
# Frequency Penalty = 0 (brak kary)
"Kot kot kot może powtarzać kot"

# Frequency Penalty = 0.5 (średnia kara)
"Kot jest zwierzęciem, które lubi mleko i zabawy"

# Frequency Penalty = 2.0 (silna kara)
"Kot to zwierzę domowe lubiące mleko, zabawy oraz polowania"
```

#### Presence Penalty
Karze za używanie tokenów, które już wystąpiły.

```python
# Presence Penalty = 0
"Lubię pizzę. Pizza jest dobra. Jem pizzę codziennie."

# Presence Penalty = 0.6
"Lubię pizzę. Ta włoska potrawa jest pyszna. Jem ją często."

# Presence Penalty = 2.0  
"Kocham pizzę. Włoskie danie zachwyca smakiem. Delektuję się regularnie."
```

### 1.3 Techniki promptowe redukujące halucynacje

#### 1. Explicit Uncertainty Instructions
```
"Jeśli nie znasz odpowiedzi, napisz 'Nie mam pewności' lub 'Nie posiadam tej informacji'.
Nie zgaduj i nie wymyślaj faktów."
```

#### 2. Source Request
```
"Dla każdego podanego faktu, podaj źródło lub zaznacz że jest to ogólna wiedza.
Format: [Fakt] (Źródło: ...)"
```

#### 3. Step-by-Step Verification
```
"Zanim odpowiesz:
1. Zidentyfikuj fakty w pytaniu
2. Sprawdź swoją wiedzę o każdym
3. Zaznacz poziom pewności (Wysoki/Średni/Niski)
4. Odpowiedz tylko używając faktów z wysoką pewnością"
```

#### 4. Fact-Checking Prompt
```
"Po wygenerowaniu odpowiedzi, przejrzyj ją i:
1. Oznacz wszystkie stwierdzenia faktyczne [F]
2. Oceń prawdopodobieństwo każdego (0-100%)
3. Usuń lub zmodyfikuj stwierdzenia <70%"
```

### 1.4 Implementacja walidacji

```python
class HallucinationDetector:
    def __init__(self):
        self.fact_patterns = [
            r'\d{4}',  # years
            r'\d+%',   # percentages
            r'\"[^\"]+\"',  # quotes
            r'według \w+',  # citations
        ]
        
    def flag_potential_hallucinations(self, text):
        flags = []
        for pattern in self.fact_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                flags.append({
                    'text': match,
                    'type': 'potential_fact',
                    'confidence': self.assess_confidence(match)
                })
        return flags
    
    def assess_confidence(self, fact):
        # Implementacja oceny pewności
        return 0.5  # placeholder
```

## 2. Kontrolowanie stylu, tonu i zakresu

### 2.1 Definiowanie stylu

#### Komponenty stylu:
1. **Formalność**
   ```
   Formalny: "Szanowni Państwo, pragnę przedstawić..."
   Neutralny: "Chciałbym przedstawić..."
   Nieformalny: "Hej, pokażę wam..."
   ```

2. **Techniczny poziom**
   ```
   Ekspert: "Implementacja algorytmu wykorzystuje O(n log n)..."
   Średni: "Program sortuje dane efektywnie..."
   Podstawowy: "Program układa rzeczy w kolejności..."
   ```

3. **Emocjonalność**
   ```
   Empatyczny: "Rozumiem, że to może być frustrujące..."
   Neutralny: "Problem został zidentyfikowany..."
   Dystansowany: "Należy rozwiązać następującą kwestię..."
   ```

### 2.2 Instrukcje stylowe

```python
style_instructions = {
    "corporate_formal": """
    Styl: Korporacyjny, profesjonalny
    - Używaj pełnych zdań i poprawnej gramatyki
    - Unikaj skrótów i kolokwializmów
    - Zwracaj się per "Państwo" lub "Pan/Pani"
    - Używaj strony biernej gdy to stosowne
    """,
    
    "friendly_casual": """
    Styl: Przyjazny, swobodny
    - Używaj prostego, zrozumiałego języka
    - Możesz używać "ty" i skrótów
    - Dodawaj emotikony gdzie stosowne 😊
    - Bądź entuzjastyczny i pozytywny
    """,
    
    "technical_documentation": """
    Styl: Techniczny, precyzyjny
    - Używaj terminologii branżowej
    - Strukturyzuj informacje (punkty, sekcje)
    - Podawaj konkretne przykłady kodu
    - Unikaj niepotrzebnych ozdobników
    """
}
```

### 2.3 Kontrola tonu

```python
class ToneController:
    def __init__(self):
        self.tone_markers = {
            'enthusiastic': ['wspaniale', 'fantastycznie', 'ekscytujące', '!'],
            'professional': ['proszę', 'uprzejmie', 'z poważaniem'],
            'empathetic': ['rozumiem', 'współczuję', 'przykro mi'],
            'authoritative': ['należy', 'wymaga się', 'obowiązkowo']
        }
    
    def apply_tone(self, prompt, tone):
        tone_instruction = f"\nTon wypowiedzi: {tone}"
        tone_examples = f"\nUżywaj słów takich jak: {', '.join(self.tone_markers.get(tone, []))}"
        return prompt + tone_instruction + tone_examples
```

### 2.4 Ograniczanie zakresu

#### Content Boundaries
```
"Ograniczenia tematyczne:
- Odpowiadaj TYLKO na pytania związane z [TEMAT]
- Jeśli pytanie wykracza poza zakres, grzecznie przekieruj
- Nie poruszaj tematów: [LISTA ZAKAZANYCH]
- Przykład przekierowania: 'Skupmy się na [TEMAT]...'"
```

#### Length Control
```python
def create_length_controlled_prompt(content, min_words=50, max_words=200):
    return f"""
    {content}
    
    Wymagania dotyczące długości:
    - Minimum {min_words} słów
    - Maximum {max_words} słów
    - Jeśli temat wymaga więcej, podsumuj kluczowe punkty
    """
```

#### Format Enforcement
```python
response_formats = {
    "bullet_points": """
    Format odpowiedzi:
    • Punkt pierwszy
    • Punkt drugi
    • Każdy punkt max 2 zdania
    """,
    
    "numbered_steps": """
    Format odpowiedzi:
    1. Krok pierwszy (co zrobić)
       - Szczegół A
       - Szczegół B
    2. Krok drugi (kolejna akcja)
    """,
    
    "qa_format": """
    Format odpowiedzi:
    P: [Pytanie]
    O: [Odpowiedź - max 3 zdania]
    """
}
```

## 3. Implementacja zabezpieczeń

### 3.1 Moderacja automatyczna

```python
class ContentModerator:
    def __init__(self):
        self.blocked_terms = set()
        self.sensitive_topics = []
        self.toxicity_threshold = 0.7
        
    def moderate_input(self, text):
        """Sprawdza input użytkownika przed wysłaniem do LLM"""
        issues = []
        
        # Sprawdzenie blocked terms
        for term in self.blocked_terms:
            if term.lower() in text.lower():
                issues.append(f"Blocked term: {term}")
        
        # Sprawdzenie prompt injection
        if self.detect_prompt_injection(text):
            issues.append("Potential prompt injection detected")
            
        # Sprawdzenie długości
        if len(text) > 10000:
            issues.append("Input too long")
            
        return issues
    
    def moderate_output(self, text):
        """Sprawdza output LLM przed zwróceniem użytkownikowi"""
        issues = []
        
        # Sprawdzenie toksyczności
        toxicity = self.calculate_toxicity(text)
        if toxicity > self.toxicity_threshold:
            issues.append(f"High toxicity: {toxicity}")
        
        # Sprawdzenie PII (Personal Identifiable Information)
        if self.detect_pii(text):
            issues.append("PII detected")
            
        return issues
    
    def detect_prompt_injection(self, text):
        """Wykrywa próby prompt injection"""
        injection_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "new instructions:",
            "system: ",
            "[[INST]]",
        ]
        return any(pattern in text.lower() for pattern in injection_patterns)
    
    def detect_pii(self, text):
        """Wykrywa dane osobowe"""
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{3}\b',
            'pesel': r'\b\d{11}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
        
        for pii_type, pattern in patterns.items():
            if re.search(pattern, text):
                return True
        return False
```

### 3.2 Filtrowanie treści

```python
class ContentFilter:
    def __init__(self):
        self.filters = {
            'violence': self.filter_violence,
            'adult': self.filter_adult_content,
            'medical': self.filter_medical_advice,
            'financial': self.filter_financial_advice,
            'legal': self.filter_legal_advice
        }
    
    def apply_filters(self, content, active_filters):
        """Aplikuje wybrane filtry do treści"""
        filtered_content = content
        applied_filters = []
        
        for filter_name in active_filters:
            if filter_name in self.filters:
                filtered_content, was_filtered = self.filters[filter_name](filtered_content)
                if was_filtered:
                    applied_filters.append(filter_name)
        
        return filtered_content, applied_filters
    
    def filter_medical_advice(self, content):
        """Filtruje porady medyczne"""
        medical_keywords = ['diagnoza', 'leczenie', 'dawkowanie', 'choroba']
        
        if any(keyword in content.lower() for keyword in medical_keywords):
            disclaimer = "\n\n⚠️ Uwaga: To nie jest porada medyczna. Skonsultuj się z lekarzem."
            return content + disclaimer, True
        
        return content, False
```

### 3.3 Walidacja odpowiedzi

```python
class ResponseValidator:
    def __init__(self):
        self.validation_rules = []
    
    def add_rule(self, rule_name, validation_func):
        """Dodaje regułę walidacji"""
        self.validation_rules.append({
            'name': rule_name,
            'func': validation_func
        })
    
    def validate(self, response):
        """Waliduje odpowiedź według wszystkich reguł"""
        validation_results = {
            'is_valid': True,
            'failed_rules': [],
            'warnings': []
        }
        
        for rule in self.validation_rules:
            result = rule['func'](response)
            if not result['passed']:
                validation_results['is_valid'] = False
                validation_results['failed_rules'].append(rule['name'])
            if 'warning' in result:
                validation_results['warnings'].append(result['warning'])
        
        return validation_results

# Przykładowe reguły walidacji
def validate_no_urls(response):
    """Sprawdza czy odpowiedź nie zawiera URL"""
    url_pattern = r'https?://\S+'
    if re.search(url_pattern, response):
        return {'passed': False, 'message': 'URLs not allowed'}
    return {'passed': True}

def validate_language(response, allowed_lang='pl'):
    """Sprawdza język odpowiedzi"""
    # Simplified - w praktyce użyj biblioteki do detekcji języka
    if allowed_lang == 'pl' and not any(char in response for char in 'ąćęłńóśźż'):
        return {'passed': False, 'warning': 'Response might not be in Polish'}
    return {'passed': True}
```

### 3.4 System bezpieczeństwa end-to-end

```python
class SecureLLMSystem:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.moderator = ContentModerator()
        self.filter = ContentFilter()
        self.validator = ResponseValidator()
        self.audit_log = []
        
    def process_request(self, user_input, user_id=None):
        """Bezpieczne przetwarzanie zapytania"""
        request_id = str(uuid.uuid4())
        
        # 1. Logowanie
        self.log_request(request_id, user_id, user_input)
        
        # 2. Moderacja inputu
        input_issues = self.moderator.moderate_input(user_input)
        if input_issues:
            return self.handle_blocked_input(input_issues)
        
        # 3. Przygotowanie bezpiecznego promptu
        safe_prompt = self.prepare_safe_prompt(user_input)
        
        # 4. Wywołanie LLM
        try:
            response = self.llm.generate(
                safe_prompt,
                temperature=0.3,  # Niska dla bezpieczeństwa
                max_tokens=500    # Limit długości
            )
        except Exception as e:
            return self.handle_llm_error(e)
        
        # 5. Moderacja outputu
        output_issues = self.moderator.moderate_output(response)
        if output_issues:
            response = self.sanitize_response(response, output_issues)
        
        # 6. Filtrowanie
        response, applied_filters = self.filter.apply_filters(
            response, 
            ['medical', 'financial', 'legal']
        )
        
        # 7. Walidacja końcowa
        validation = self.validator.validate(response)
        if not validation['is_valid']:
            response = self.handle_validation_failure(response, validation)
        
        # 8. Logowanie odpowiedzi
        self.log_response(request_id, response, applied_filters)
        
        return {
            'response': response,
            'metadata': {
                'filtered': len(applied_filters) > 0,
                'warnings': validation.get('warnings', [])
            }
        }
    
    def prepare_safe_prompt(self, user_input):
        """Przygotowuje bezpieczny prompt"""
        return f"""
        Jesteś pomocnym asystentem. Przestrzegaj następujących zasad:
        1. Nie generuj treści szkodliwych, obraźliwych lub nielegalnych
        2. Nie udostępniaj danych osobowych
        3. Nie udzielaj porad medycznych, prawnych ani finansowych
        4. Bądź pomocny i profesjonalny
        
        Zapytanie użytkownika: {user_input}
        """
```

## 4. Praktyczne przykłady implementacji

### 4.1 Bezpieczny chatbot korporacyjny

```python
class CorporateChatbot:
    def __init__(self):
        self.system_prompt = """
        Jesteś asystentem korporacyjnym firmy TechCorp.
        
        ZASADY:
        1. Odpowiadaj TYLKO na pytania związane z firmą
        2. Używaj formalnego, profesjonalnego tonu
        3. Nie ujawniaj informacji poufnych
        4. Przekierowuj pytania HR do działu kadr
        5. Nie komentuj konkurencji
        
        DOZWOLONE TEMATY:
        - Produkty i usługi firmy
        - Godziny otwarcia i kontakt
        - Ogólne informacje o firmie
        - Publiczne osiągnięcia
        
        NIEDOZWOLONE:
        - Dane finansowe szczegółowe
        - Informacje o pracownikach
        - Plany strategiczne
        - Krytyka konkurencji
        """
        
        self.topic_redirects = {
            'hr': "W sprawach kadrowych proszę kontaktować się z działem HR: hr@techcorp.com",
            'finance': "Szczegółowe dane finansowe dostępne są w raportach giełdowych",
            'competitor': "Skupiamy się na naszych produktach i wartości dla klientów"
        }
```

### 4.2 System analizy sentymentu z kontrolą jakości

```python
class SentimentAnalyzer:
    def __init__(self):
        self.calibration_examples = {
            'positive': [
                "Świetny produkt! Polecam każdemu.",
                "Jestem zachwycony jakością obsługi."
            ],
            'neutral': [
                "Produkt spełnia swoją funkcję.",
                "Dostawa przyszła na czas."
            ],
            'negative': [
                "Rozczarowanie. Nie polecam.",
                "Słaba jakość za tę cenę."
            ]
        }
    
    def analyze_with_confidence(self, text):
        prompt = f"""
        Przeanalizuj sentyment następującego tekstu.
        
        Przykłady kalibracyjne:
        POZYTYWNY: {self.calibration_examples['positive'][0]}
        NEUTRALNY: {self.calibration_examples['neutral'][0]}
        NEGATYWNY: {self.calibration_examples['negative'][0]}
        
        Tekst do analizy: "{text}"
        
        Odpowiedz w formacie:
        Sentyment: [POZYTYWNY/NEUTRALNY/NEGATYWNY]
        Pewność: [0-100]%
        Kluczowe frazy: [lista fraz wpływających na ocenę]
        """
        
        return self.parse_sentiment_response(prompt)
```

### 4.3 Generator raportów z kontrolą faktów

```python
class ReportGenerator:
    def __init__(self):
        self.fact_check_prompt = """
        Generując raport, stosuj następujące zasady:
        
        1. Dla każdego stwierdzenia liczbowego dodaj [FAKT] lub [SZACUNEK]
        2. Unikaj stwierdzeń bez podstaw - użyj "prawdopodobnie", "około", "szacunkowo"
        3. Jeśli brakuje danych, napisz [BRAK DANYCH]
        4. Używaj zakresu zamiast konkretnych liczb gdy nie masz pewności
        
        Format:
        - Użyj nagłówków dla sekcji
        - Punktuj kluczowe wnioski
        - Dodaj sekcję "Zastrzeżenia" na końcu
        """
    
    def generate_report(self, data, report_type):
        enhanced_prompt = f"""
        {self.fact_check_prompt}
        
        Typ raportu: {report_type}
        Dane wejściowe: {data}
        
        Wygeneruj raport przestrzegając wszystkich zasad bezpieczeństwa.
        """
        
        report = self.llm.generate(enhanced_prompt, temperature=0.2)
        return self.post_process_report(report)
    
    def post_process_report(self, report):
        """Dodaje dodatkowe zabezpieczenia do raportu"""
        # Dodaj timestamp
        report = f"Raport wygenerowany: {datetime.now()}\n\n{report}"
        
        # Dodaj disclaimer
        disclaimer = """
        
        ---
        UWAGA: Ten raport został wygenerowany automatycznie. 
        Zaleca się weryfikację kluczowych danych przed podjęciem decyzji biznesowych.
        """
        
        return report + disclaimer
```

## 5. Ćwiczenia praktyczne

### Ćwiczenie 1: Kalibracja parametrów
1. Wybierz zadanie (np. generowanie opisu produktu)
2. Przetestuj różne kombinacje:
   - Temperature: 0, 0.3, 0.7, 1.0
   - Top-p: 0.5, 0.9, 1.0
3. Oceń wyniki pod kątem:
   - Kreatywności
   - Spójności
   - Faktualności
4. Znajdź optymalne ustawienia

### Ćwiczenie 2: Detekcja halucynacji
1. Stwórz prompt proszący o:
   - Cytat z nieistniejącej książki
   - Dane statystyczne z przyszłości
   - Szczegóły nieistniejącego wydarzenia
2. Analizuj jak model reaguje
3. Dodaj instrukcje anti-halucynacyjne
4. Porównaj wyniki

### Ćwiczenie 3: Implementacja moderatora
1. Zaimplementuj prosty system moderacji:
   - Lista zakazanych słów
   - Detekcja prompt injection
   - Sprawdzanie długości
2. Przetestuj na różnych inputach
3. Dodaj logowanie zdarzeń
4. Oceń false positives/negatives

### Ćwiczenie 4: Kontrola stylu
1. Wybierz temat (np. "Sztuczna inteligencja")
2. Wygeneruj ten sam content w 4 stylach:
   - Akademicki
   - Biznesowy
   - Casual/blog
   - Dla dzieci
3. Analizuj różnice
4. Stwórz "style guide" dla każdego

## 6. Case study: Bezpieczny chatbot medyczny

```python
class MedicalChatbot:
    def __init__(self):
        self.strict_prompt = """
        KRYTYCZNE: Jesteś asystentem informacyjnym, NIE lekarzem.
        
        ABSOLUTNIE ZAKAZANE:
        - Diagnozowanie chorób
        - Zalecanie leków lub dawkowania  
        - Interpretacja wyników badań
        - Odradzanie wizyty u lekarza
        
        DOZWOLONE:
        - Ogólne informacje o zdrowiu
        - Wyjaśnianie terminów medycznych
        - Informacje o zdrowym stylu życia
        - Zachęcanie do konsultacji z lekarzem
        
        KAŻDA odpowiedź MUSI zawierać:
        "To nie jest porada medyczna. Skonsultuj się z lekarzem."
        """
    
    def process_medical_query(self, query):
        # Sprawdź czy pytanie dotyczy konkretnych objawów
        symptom_keywords = ['boli', 'bolę', 'objawy', 'choroba', 'leczenie']
        
        if any(keyword in query.lower() for keyword in symptom_keywords):
            return self.redirect_to_doctor(query)
        
        # Standardowe przetwarzanie z disclaimerem
        response = self.generate_safe_response(query)
        return self.add_medical_disclaimer(response)
    
    def redirect_to_doctor(self, query):
        return """
        Opisywane objawy wymagają profesjonalnej oceny medycznej.
        
        Co możesz zrobić:
        1. Skontaktuj się z lekarzem pierwszego kontaktu
        2. W nagłych przypadkach - zadzwoń na 112
        3. Teleporada - wiele przychodni oferuje konsultacje online
        
        ⚠️ To nie jest porada medyczna. Tylko lekarz może postawić diagnozę.
        """
```

## 7. Metryki i monitoring

### 7.1 KPIs bezpieczeństwa

```python
class SafetyMetrics:
    def __init__(self):
        self.metrics = {
            'hallucination_rate': 0,
            'blocked_requests': 0,
            'filtered_responses': 0,
            'user_complaints': 0,
            'false_positives': 0
        }
    
    def calculate_safety_score(self):
        """Oblicza ogólny wskaźnik bezpieczeństwa"""
        weights = {
            'hallucination_rate': -0.3,
            'blocked_requests': -0.1,
            'filtered_responses': -0.1,
            'user_complaints': -0.4,
            'false_positives': -0.1
        }
        
        score = 100
        for metric, value in self.metrics.items():
            score += weights.get(metric, 0) * value
            
        return max(0, min(100, score))
```

### 7.2 Dashboard monitoringu

```python
def create_safety_dashboard():
    """Tworzy dashboard do monitorowania bezpieczeństwa"""
    return {
        'real_time_metrics': {
            'active_sessions': get_active_sessions(),
            'requests_per_minute': calculate_rpm(),
            'current_alerts': get_active_alerts()
        },
        'daily_stats': {
            'total_requests': get_daily_requests(),
            'blocked_percentage': calculate_block_rate(),
            'average_response_time': get_avg_response_time(),
            'user_satisfaction': get_satisfaction_score()
        },
        'top_issues': {
            'blocked_patterns': get_top_blocked_patterns(),
            'common_hallucinations': get_hallucination_patterns(),
            'user_complaints': get_complaint_categories()
        }
    }
```

## 8. Najlepsze praktyki

### 8.1 Checklist bezpieczeństwa

- [ ] Zdefiniowano system prompt z jasnymi ograniczeniami
- [ ] Zaimplementowano walidację inputu
- [ ] Skonfigurowano moderację outputu
- [ ] Ustawiono odpowiednie parametry (temperature, max_tokens)
- [ ] Dodano disclaimery gdzie potrzebne
- [ ] Zaimplementowano logowanie
- [ ] Przygotowano obsługę błędów
- [ ] Przetestowano edge cases
- [ ] Skonfigurowano monitoring
- [ ] Przygotowano procedury eskalacji

### 8.2 Continuous Improvement

```python
class SafetyImprovement:
    def analyze_incidents(self, timeframe='7d'):
        """Analizuje incydenty bezpieczeństwa"""
        incidents = self.get_incidents(timeframe)
        
        patterns = {
            'prompt_injection': [],
            'hallucinations': [],
            'toxic_content': [],
            'pii_leaks': []
        }
        
        for incident in incidents:
            category = self.categorize_incident(incident)
            patterns[category].append(incident)
        
        return self.generate_improvement_recommendations(patterns)
```

## 9. Podsumowanie

1. **Bezpieczeństwo to proces ciągły, nie jednorazowa konfiguracja**
2. **Parametry modelu znacząco wpływają na jakość i bezpieczeństwo**
3. **Wielowarstwowe zabezpieczenia są najskuteczniejsze**
4. **Monitoring i analiza są kluczowe dla utrzymania bezpieczeństwa**
5. **Transparentność (disclaimery) buduje zaufanie użytkowników**

## 10. Zadanie praktyczne

Zaprojektuj i zaimplementuj bezpieczny system chatbota dla banku:

1. **Wymagania**:
   - Obsługa pytań o produkty bankowe
   - Brak dostępu do danych klientów
   - Przekierowanie do konsultanta gdy trzeba
   - Detekcja prób wyłudzenia informacji

2. **Implementacja**:
   - System prompt z ograniczeniami
   - Moderacja input/output
   - Filtrowanie danych wrażliwych
   - System logowania

3. **Testy**:
   - 10 normalnych zapytań
   - 5 prób prompt injection
   - 5 pytań o dane wrażliwe
   - Analiza wyników

## Materiały dodatkowe

- "Red Teaming Language Models" - Anthropic
- "Best Practices for Deploying Language Models" - Google
- "Content Moderation with LLMs" - OpenAI
- "AI Safety Fundamentals" - DeepMind