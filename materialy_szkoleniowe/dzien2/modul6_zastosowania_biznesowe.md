# Moduł 6: Praktyczne zastosowania kontrolowania LLM w biznesie

## Cel modułu
Po zakończeniu tego modułu uczestnik będzie:
- Projektował i implementował chatboty do obsługi klienta
- Tworzył systemy generowania raportów i dokumentacji
- Implementował personalizację treści
- Rozumiał architekturę i integrację systemów LLM w środowisku produkcyjnym

## 1. Chatboty i systemy automatycznej obsługi klienta

### 1.1 Architektura systemu chatbota korporacyjnego

```python
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import json

class EnterpriseCustomerServiceBot:
    def __init__(self, config: Dict):
        self.llm_client = config['llm_client']
        self.crm_integration = config['crm_integration']
        self.knowledge_base = config['knowledge_base']
        self.escalation_rules = config['escalation_rules']
        self.sentiment_analyzer = config['sentiment_analyzer']
        
        self.system_prompt = """
        Jesteś profesjonalnym konsultantem obsługi klienta firmy TechCorp.
        
        TWOJE ZADANIA:
        1. Pomagaj klientom w rozwiązywaniu problemów
        2. Udzielaj informacji o produktach i usługach
        3. Zbieraj informacje potrzebne do rozwiązania sprawy
        4. Eskaluj złożone sprawy do odpowiednich działów
        
        ZASADY:
        - Bądź uprzejmy i profesjonalny
        - Używaj prostego, zrozumiałego języka
        - Nie obiecuj rzeczy, których nie możesz zagwarantować
        - Zawsze weryfikuj tożsamość klienta przed udostępnieniem wrażliwych informacji
        - Dokumentuj wszystkie istotne informacje
        
        OGRANICZENIA:
        - Nie udzielaj informacji o danych innych klientów
        - Nie modyfikuj danych w systemie bez autoryzacji
        - Nie ujawniaj wewnętrznych procesów firmy
        """
        
    async def handle_conversation(self, customer_id: str, message: str, context: Dict):
        """Główna funkcja obsługująca konwersację"""
        # 1. Analiza sentymentu i intencji
        sentiment = await self.analyze_customer_sentiment(message)
        intent = await self.detect_intent(message, context)
        
        # 2. Pobranie kontekstu klienta
        customer_data = await self.get_customer_context(customer_id)
        
        # 3. Sprawdzenie czy potrzebna eskalacja
        if self.needs_escalation(sentiment, intent, customer_data):
            return await self.escalate_to_human(customer_id, message, context)
        
        # 4. Przygotowanie odpowiedzi
        response = await self.generate_response(
            message, 
            customer_data, 
            intent, 
            context
        )
        
        # 5. Post-processing i walidacja
        validated_response = await self.validate_and_enhance_response(response)
        
        # 6. Logowanie interakcji
        await self.log_interaction(customer_id, message, validated_response)
        
        return validated_response
    
    async def analyze_customer_sentiment(self, message: str) -> Dict:
        """Analizuje nastrój klienta"""
        sentiment_score = await self.sentiment_analyzer.analyze(message)
        
        return {
            'score': sentiment_score,
            'category': self.categorize_sentiment(sentiment_score),
            'urgency': self.calculate_urgency(sentiment_score, message)
        }
    
    def categorize_sentiment(self, score: float) -> str:
        if score < -0.5:
            return 'very_negative'
        elif score < -0.1:
            return 'negative'
        elif score < 0.1:
            return 'neutral'
        elif score < 0.5:
            return 'positive'
        else:
            return 'very_positive'
    
    async def detect_intent(self, message: str, context: Dict) -> Dict:
        """Wykrywa intencję klienta"""
        intent_prompt = f"""
        Przeanalizuj wiadomość klienta i określ intencję.
        
        Wiadomość: "{message}"
        Kontekst rozmowy: {json.dumps(context, ensure_ascii=False)}
        
        Możliwe intencje:
        - product_inquiry: Pytanie o produkt
        - technical_support: Problem techniczny
        - billing_issue: Problem z płatnością
        - complaint: Reklamacja
        - general_question: Ogólne pytanie
        - order_status: Status zamówienia
        - account_management: Zarządzanie kontem
        
        Odpowiedz w formacie JSON:
        {{
            "primary_intent": "nazwa_intencji",
            "confidence": 0.0-1.0,
            "entities": {{}}
        }}
        """
        
        response = await self.llm_client.generate(intent_prompt, temperature=0.1)
        return json.loads(response)
    
    async def get_customer_context(self, customer_id: str) -> Dict:
        """Pobiera kontekst klienta z CRM"""
        customer_data = await self.crm_integration.get_customer(customer_id)
        
        # Dodaj historię ostatnich interakcji
        recent_interactions = await self.crm_integration.get_recent_interactions(
            customer_id, 
            limit=5
        )
        
        # Dodaj otwarte sprawy
        open_tickets = await self.crm_integration.get_open_tickets(customer_id)
        
        return {
            'profile': customer_data,
            'recent_interactions': recent_interactions,
            'open_tickets': open_tickets,
            'customer_value': self.calculate_customer_value(customer_data),
            'preferred_language': customer_data.get('language', 'pl')
        }
    
    def needs_escalation(self, sentiment: Dict, intent: Dict, customer_data: Dict) -> bool:
        """Sprawdza czy potrzebna jest eskalacja do człowieka"""
        # Eskaluj bardzo negatywne nastroje
        if sentiment['category'] == 'very_negative':
            return True
            
        # Eskaluj VIP klientów z problemami
        if customer_data['customer_value'] == 'vip' and sentiment['category'] == 'negative':
            return True
            
        # Eskaluj określone intencje
        escalation_intents = ['legal_threat', 'data_breach', 'serious_complaint']
        if intent['primary_intent'] in escalation_intents:
            return True
            
        # Eskaluj po określonej liczbie interakcji bez rozwiązania
        if len(customer_data['recent_interactions']) > 5:
            return True
            
        return False
    
    async def generate_response(self, message: str, customer_data: Dict, 
                               intent: Dict, context: Dict) -> str:
        """Generuje odpowiedź używając LLM"""
        # Przygotuj dane z knowledge base
        relevant_info = await self.knowledge_base.search(
            query=message,
            intent=intent['primary_intent'],
            limit=3
        )
        
        prompt = f"""
        {self.system_prompt}
        
        DANE KLIENTA:
        - ID: {customer_data['profile']['id']}
        - Typ: {customer_data['customer_value']}
        - Otwarte sprawy: {len(customer_data['open_tickets'])}
        
        KONTEKST:
        {json.dumps(context, ensure_ascii=False, indent=2)}
        
        INFORMACJE Z BAZY WIEDZY:
        {self._format_knowledge_base_info(relevant_info)}
        
        WIADOMOŚĆ KLIENTA: "{message}"
        
        WYKRYTA INTENCJA: {intent['primary_intent']} (pewność: {intent['confidence']})
        
        Wygeneruj odpowiednią odpowiedź.
        """
        
        response = await self.llm_client.generate(
            prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        return response
    
    def _format_knowledge_base_info(self, info: List[Dict]) -> str:
        """Formatuje informacje z bazy wiedzy"""
        formatted = []
        for item in info:
            formatted.append(f"- {item['title']}: {item['content']}")
        return "\n".join(formatted)
    
    async def validate_and_enhance_response(self, response: str) -> str:
        """Waliduje i ulepsza odpowiedź"""
        # Sprawdź czy nie zawiera informacji wrażliwych
        if self.contains_sensitive_info(response):
            response = self.redact_sensitive_info(response)
            
        # Dodaj personalizację
        response = self.add_personalization(response)
        
        # Dodaj call-to-action jeśli potrzebne
        response = self.add_call_to_action(response)
        
        return response
    
    async def escalate_to_human(self, customer_id: str, message: str, context: Dict) -> str:
        """Eskaluje rozmowę do konsultanta"""
        # Znajdź odpowiedniego konsultanta
        agent = await self.find_available_agent(customer_id, context)
        
        # Przygotuj podsumowanie dla konsultanta
        summary = await self.prepare_escalation_summary(customer_id, message, context)
        
        # Przekaż rozmowę
        ticket_id = await self.crm_integration.create_escalation_ticket(
            customer_id=customer_id,
            agent_id=agent['id'],
            summary=summary,
            priority='high'
        )
        
        return f"""
        Rozumiem, że ta sprawa wymaga szczególnej uwagi. 
        Przekazuję Pana/Pani sprawę do specjalisty {agent['name']}, 
        który skontaktuje się z Panem/Panią w ciągu 15 minut.
        
        Numer Pana/Pani zgłoszenia to: {ticket_id}
        
        Czy mogę pomóc w czymś jeszcze w międzyczasie?
        """
```

### 1.2 Integracja z systemami CRM

```python
class CRMIntegration:
    def __init__(self, crm_config: Dict):
        self.api_endpoint = crm_config['endpoint']
        self.api_key = crm_config['api_key']
        self.cache = {}
        
    async def sync_customer_data(self, customer_id: str) -> Dict:
        """Synchronizuje dane klienta z CRM"""
        # Sprawdź cache
        if customer_id in self.cache:
            cached_data = self.cache[customer_id]
            if self._is_cache_valid(cached_data):
                return cached_data['data']
        
        # Pobierz z API
        customer_data = await self._fetch_from_api(f'/customers/{customer_id}')
        
        # Wzbogać dane
        enriched_data = await self._enrich_customer_data(customer_data)
        
        # Cache
        self.cache[customer_id] = {
            'data': enriched_data,
            'timestamp': datetime.now()
        }
        
        return enriched_data
    
    async def log_interaction(self, interaction_data: Dict):
        """Loguje interakcję w CRM"""
        payload = {
            'customer_id': interaction_data['customer_id'],
            'channel': 'chatbot',
            'timestamp': interaction_data['timestamp'],
            'messages': interaction_data['messages'],
            'sentiment': interaction_data['sentiment'],
            'resolution_status': interaction_data['resolution_status'],
            'tags': self._generate_tags(interaction_data)
        }
        
        response = await self._post_to_api('/interactions', payload)
        return response
    
    async def create_lead_from_conversation(self, conversation_data: Dict) -> str:
        """Tworzy lead na podstawie rozmowy"""
        lead_extractor_prompt = f"""
        Przeanalizuj rozmowę i wyodrębnij informacje o potencjalnym leadzie.
        
        Rozmowa:
        {json.dumps(conversation_data['messages'], ensure_ascii=False)}
        
        Wyodrębnij:
        - Zainteresowane produkty/usługi
        - Budżet (jeśli wspomniany)
        - Termin realizacji
        - Poziom zainteresowania (1-10)
        - Następne kroki
        
        Format JSON.
        """
        
        lead_info = await self.llm_client.generate(lead_extractor_prompt, temperature=0.1)
        lead_data = json.loads(lead_info)
        
        # Stwórz lead w CRM
        lead_id = await self._post_to_api('/leads', {
            'source': 'chatbot',
            'customer_id': conversation_data['customer_id'],
            'data': lead_data,
            'assigned_to': self._find_best_salesperson(lead_data)
        })
        
        return lead_id
```

### 1.3 Wielojęzyczność i lokalizacja

```python
class MultilingualChatbot:
    def __init__(self):
        self.supported_languages = ['pl', 'en', 'de', 'fr', 'es']
        self.language_models = {}
        self.translation_cache = {}
        
    async def detect_language(self, text: str) -> str:
        """Wykrywa język wiadomości"""
        detection_prompt = f"""
        Wykryj język następującego tekstu: "{text}"
        
        Odpowiedz tylko kodem języka (pl, en, de, fr, es).
        """
        
        language = await self.llm_client.generate(detection_prompt, temperature=0)
        return language.strip().lower()
    
    async def handle_multilingual_request(self, message: str, preferred_lang: str = None):
        """Obsługuje żądanie w wielu językach"""
        # Wykryj język wiadomości
        detected_lang = await self.detect_language(message)
        
        # Określ język odpowiedzi
        response_lang = preferred_lang or detected_lang
        
        # Jeśli potrzeba, przetłumacz na język roboczy (np. angielski)
        if detected_lang not in ['pl', 'en']:
            working_message = await self.translate(message, detected_lang, 'en')
        else:
            working_message = message
        
        # Generuj odpowiedź
        response = await self.generate_response(working_message)
        
        # Przetłumacz odpowiedź na język docelowy
        if response_lang not in ['pl', 'en']:
            final_response = await self.translate(response, 'pl', response_lang)
        else:
            final_response = response
            
        return {
            'response': final_response,
            'detected_language': detected_lang,
            'response_language': response_lang
        }
    
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Tłumaczy tekst między językami"""
        # Sprawdź cache
        cache_key = f"{source_lang}_{target_lang}_{hash(text)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        translation_prompt = f"""
        Przetłumacz następujący tekst z {source_lang} na {target_lang}.
        Zachowaj ton i styl wypowiedzi.
        
        Tekst: "{text}"
        
        Tłumaczenie:
        """
        
        translation = await self.llm_client.generate(translation_prompt, temperature=0.1)
        
        # Cache
        self.translation_cache[cache_key] = translation
        
        return translation
    
    def get_localized_responses(self, response_type: str, language: str) -> Dict:
        """Zwraca zlokalizowane odpowiedzi"""
        localized_responses = {
            'pl': {
                'greeting': 'Dzień dobry! Jak mogę pomóc?',
                'wait': 'Proszę chwilę poczekać, sprawdzam...',
                'escalation': 'Przekazuję sprawę do konsultanta.',
                'closing': 'Dziękuję za kontakt. Miłego dnia!'
            },
            'en': {
                'greeting': 'Hello! How can I help you?',
                'wait': 'Please wait a moment while I check...',
                'escalation': 'I\'m transferring you to a specialist.',
                'closing': 'Thank you for contacting us. Have a great day!'
            },
            'de': {
                'greeting': 'Guten Tag! Wie kann ich Ihnen helfen?',
                'wait': 'Bitte warten Sie einen Moment...',
                'escalation': 'Ich verbinde Sie mit einem Spezialisten.',
                'closing': 'Vielen Dank für Ihre Anfrage. Schönen Tag noch!'
            }
        }
        
        return localized_responses.get(language, localized_responses['en']).get(response_type)
```

## 2. Generowanie raportów, analiz i dokumentacji technicznej

### 2.1 System automatycznego generowania raportów

```python
class AutomatedReportGenerator:
    def __init__(self):
        self.templates = {}
        self.data_sources = {}
        self.quality_checker = QualityChecker()
        
    async def generate_business_report(self, report_type: str, parameters: Dict) -> Dict:
        """Generuje raport biznesowy"""
        # 1. Pobierz dane
        raw_data = await self.collect_data(report_type, parameters)
        
        # 2. Analizuj dane
        analysis = await self.analyze_data(raw_data, report_type)
        
        # 3. Generuj treść raportu
        report_content = await self.generate_content(analysis, report_type)
        
        # 4. Formatuj i wizualizuj
        formatted_report = await self.format_report(report_content, analysis)
        
        # 5. Kontrola jakości
        validated_report = await self.quality_checker.validate_report(formatted_report)
        
        return validated_report
    
    async def collect_data(self, report_type: str, parameters: Dict) -> Dict:
        """Zbiera dane z różnych źródeł"""
        data_requirements = self.get_data_requirements(report_type)
        collected_data = {}
        
        for source_name, requirements in data_requirements.items():
            source = self.data_sources[source_name]
            data = await source.fetch_data(
                requirements['query'],
                parameters['date_range'],
                requirements['filters']
            )
            collected_data[source_name] = data
            
        return collected_data
    
    async def analyze_data(self, raw_data: Dict, report_type: str) -> Dict:
        """Analizuje dane i generuje insights"""
        analysis_prompt = f"""
        Przeanalizuj następujące dane biznesowe dla raportu typu: {report_type}
        
        Dane:
        {json.dumps(raw_data, ensure_ascii=False, indent=2)}
        
        Przeprowadź analizę obejmującą:
        1. Kluczowe trendy i wzorce
        2. Porównanie do poprzedniego okresu
        3. Anomalie i wartości odstające
        4. Prognozy na podstawie trendów
        5. Rekomendacje biznesowe
        
        Odpowiedź w formacie JSON z sekcjami: trends, comparisons, anomalies, forecasts, recommendations
        """
        
        analysis_result = await self.llm_client.generate(
            analysis_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        return json.loads(analysis_result)
    
    async def generate_content(self, analysis: Dict, report_type: str) -> Dict:
        """Generuje treść raportu"""
        template = self.templates[report_type]
        
        sections = {}
        for section_name, section_template in template['sections'].items():
            section_prompt = f"""
            Wygeneruj sekcję raportu: {section_name}
            
            Template: {section_template['template']}
            Dane do wykorzystania: {analysis.get(section_template['data_key'], {})}
            
            Wymagania:
            - Styl: {section_template['style']}
            - Długość: {section_template['length']} słów
            - Zawrzyj: {section_template['must_include']}
            """
            
            section_content = await self.llm_client.generate(
                section_prompt,
                temperature=0.3
            )
            
            sections[section_name] = section_content
            
        return {
            'title': template['title'].format(**analysis),
            'sections': sections,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': report_type,
                'data_period': analysis.get('period')
            }
        }
    
    async def format_report(self, content: Dict, analysis: Dict) -> Dict:
        """Formatuje raport z wykresami i tabelami"""
        formatted_report = {
            'title': content['title'],
            'executive_summary': self.generate_executive_summary(content, analysis),
            'sections': [],
            'visualizations': [],
            'tables': []
        }
        
        # Generuj wykresy
        for chart_config in self.get_chart_requirements(content['metadata']['report_type']):
            chart = await self.generate_chart(
                data=analysis[chart_config['data_source']],
                chart_type=chart_config['type'],
                title=chart_config['title']
            )
            formatted_report['visualizations'].append(chart)
        
        # Formatuj sekcje
        for section_name, section_content in content['sections'].items():
            formatted_section = {
                'title': section_name.replace('_', ' ').title(),
                'content': section_content,
                'subsections': self.extract_subsections(section_content)
            }
            formatted_report['sections'].append(formatted_section)
        
        # Generuj tabele
        for table_config in self.get_table_requirements(content['metadata']['report_type']):
            table = self.generate_table(
                data=analysis[table_config['data_source']],
                columns=table_config['columns'],
                formatting=table_config['formatting']
            )
            formatted_report['tables'].append(table)
        
        return formatted_report
    
    def generate_executive_summary(self, content: Dict, analysis: Dict) -> str:
        """Generuje streszczenie wykonawcze"""
        summary_prompt = f"""
        Stwórz zwięzłe streszczenie wykonawcze (executive summary) na podstawie raportu.
        
        Kluczowe informacje:
        - Główne trendy: {analysis['trends']}
        - Najważniejsze rekomendacje: {analysis['recommendations']}
        - Krytyczne wskaźniki: {analysis.get('kpis', {})}
        
        Wymagania:
        - Maksymalnie 200 słów
        - Skup się na tym co najważniejsze dla zarządu
        - Użyj liczb i konkretów
        - Zakończ kluczową rekomendacją
        """
        
        return self.llm_client.generate(summary_prompt, temperature=0.2)
```

### 2.2 Generator dokumentacji technicznej

```python
class TechnicalDocumentationGenerator:
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.example_generator = ExampleGenerator()
        
    async def generate_api_documentation(self, api_spec: Dict) -> str:
        """Generuje dokumentację API"""
        doc_sections = []
        
        # 1. Overview
        overview = await self.generate_overview(api_spec)
        doc_sections.append(overview)
        
        # 2. Authentication
        auth_section = await self.generate_auth_documentation(api_spec['auth'])
        doc_sections.append(auth_section)
        
        # 3. Endpoints
        for endpoint in api_spec['endpoints']:
            endpoint_doc = await self.generate_endpoint_documentation(endpoint)
            doc_sections.append(endpoint_doc)
        
        # 4. Examples
        examples = await self.generate_code_examples(api_spec)
        doc_sections.append(examples)
        
        # 5. Error handling
        error_docs = await self.generate_error_documentation(api_spec['errors'])
        doc_sections.append(error_docs)
        
        return self.compile_documentation(doc_sections)
    
    async def generate_endpoint_documentation(self, endpoint: Dict) -> str:
        """Dokumentuje pojedynczy endpoint"""
        doc_prompt = f"""
        Wygeneruj dokumentację dla endpointu API:
        
        Endpoint: {endpoint['path']}
        Metoda: {endpoint['method']}
        Opis: {endpoint['description']}
        Parametry: {json.dumps(endpoint['parameters'], indent=2)}
        Response: {json.dumps(endpoint['response_schema'], indent=2)}
        
        Struktura dokumentacji:
        1. Opis endpointu
        2. Parametry (tabela)
        3. Przykład żądania
        4. Przykład odpowiedzi
        5. Możliwe błędy
        6. Uwagi dotyczące użycia
        
        Format: Markdown
        """
        
        documentation = await self.llm_client.generate(doc_prompt, temperature=0.1)
        
        # Dodaj przykłady kodu w różnych językach
        code_examples = await self.example_generator.generate_examples(
            endpoint,
            languages=['python', 'javascript', 'curl']
        )
        
        return documentation + "\n\n" + code_examples
    
    async def generate_code_documentation(self, code_path: str) -> str:
        """Generuje dokumentację kodu"""
        # Analizuj kod
        code_analysis = await self.code_analyzer.analyze(code_path)
        
        doc_sections = {
            'overview': await self.generate_module_overview(code_analysis),
            'classes': await self.document_classes(code_analysis['classes']),
            'functions': await self.document_functions(code_analysis['functions']),
            'usage_examples': await self.generate_usage_examples(code_analysis),
            'dependencies': self.document_dependencies(code_analysis['imports'])
        }
        
        return self.format_code_documentation(doc_sections)
    
    async def document_classes(self, classes: List[Dict]) -> str:
        """Dokumentuje klasy"""
        class_docs = []
        
        for class_info in classes:
            doc_prompt = f"""
            Wygeneruj dokumentację dla klasy Python:
            
            Nazwa: {class_info['name']}
            Docstring: {class_info.get('docstring', 'Brak')}
            Metody: {[m['name'] for m in class_info['methods']]}
            Atrybuty: {class_info['attributes']}
            
            Zawrzyj:
            1. Opis klasy i jej przeznaczenie
            2. Parametry konstruktora
            3. Publiczne metody (z opisami)
            4. Przykład użycia
            5. Powiązane klasy/interfejsy
            """
            
            class_doc = await self.llm_client.generate(doc_prompt, temperature=0.1)
            class_docs.append(class_doc)
            
        return "\n\n".join(class_docs)
```

### 2.3 System analizy danych z generowaniem insights

```python
class DataAnalysisReportGenerator:
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_engine = VisualizationEngine()
        
    async def generate_data_analysis_report(self, dataset: pd.DataFrame, 
                                          analysis_type: str) -> Dict:
        """Generuje raport analizy danych"""
        # 1. Podstawowa analiza statystyczna
        basic_stats = self.statistical_analyzer.compute_basic_stats(dataset)
        
        # 2. Analiza zaawansowana
        advanced_analysis = await self.perform_advanced_analysis(dataset, analysis_type)
        
        # 3. Generowanie insights
        insights = await self.generate_insights(basic_stats, advanced_analysis)
        
        # 4. Wizualizacje
        visualizations = await self.create_visualizations(dataset, insights)
        
        # 5. Generowanie narracji
        narrative = await self.generate_narrative(insights, visualizations)
        
        return {
            'summary': self.create_executive_summary(insights),
            'detailed_analysis': narrative,
            'visualizations': visualizations,
            'recommendations': await self.generate_recommendations(insights),
            'technical_appendix': self.create_technical_appendix(basic_stats, advanced_analysis)
        }
    
    async def generate_insights(self, basic_stats: Dict, advanced_analysis: Dict) -> List[Dict]:
        """Generuje insights na podstawie analizy"""
        insights_prompt = f"""
        Na podstawie poniższej analizy danych, wygeneruj kluczowe insights biznesowe:
        
        Statystyki podstawowe:
        {json.dumps(basic_stats, indent=2)}
        
        Analiza zaawansowana:
        {json.dumps(advanced_analysis, indent=2)}
        
        Dla każdego insight podaj:
        1. Tytuł (krótki, chwytliwy)
        2. Opis odkrycia
        3. Wpływ na biznes
        4. Poziom pewności (1-10)
        5. Rekomendowane działania
        
        Szukaj:
        - Nietypowych wzorców
        - Silnych korelacji
        - Trendów czasowych
        - Anomalii
        - Możliwości optymalizacji
        """
        
        insights_json = await self.llm_client.generate(
            insights_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        insights = json.loads(insights_json)
        
        # Waliduj i rankuj insights
        validated_insights = []
        for insight in insights:
            if self.validate_insight(insight, basic_stats, advanced_analysis):
                insight['score'] = self.calculate_insight_score(insight)
                validated_insights.append(insight)
        
        # Sortuj według ważności
        validated_insights.sort(key=lambda x: x['score'], reverse=True)
        
        return validated_insights
    
    def validate_insight(self, insight: Dict, stats: Dict, analysis: Dict) -> bool:
        """Waliduje czy insight jest poprawny"""
        # Sprawdź czy liczby się zgadzają
        if 'numbers' in insight:
            for number in insight['numbers']:
                if not self.verify_number_in_data(number, stats, analysis):
                    return False
        
        # Sprawdź logiczną spójność
        if insight['confidence'] < 5 and insight['impact'] == 'high':
            return False
            
        return True
    
    async def generate_narrative(self, insights: List[Dict], visualizations: List[Dict]) -> str:
        """Generuje narrację łączącą insights w spójną historię"""
        narrative_prompt = f"""
        Stwórz spójną narrację biznesową łączącą poniższe insights w logiczną historię.
        
        Insights (w kolejności ważności):
        {json.dumps(insights[:5], indent=2)}
        
        Dostępne wizualizacje:
        {[v['title'] for v in visualizations]}
        
        Struktura narracji:
        1. Wprowadzenie - kontekst analizy
        2. Główne odkrycia - 3 najważniejsze insights
        3. Szczegółowa analiza - pozostałe insights z odwołaniami do wykresów
        4. Wnioski i implikacje biznesowe
        
        Styl: profesjonalny ale przystępny, używaj konkretnych liczb
        """
        
        narrative = await self.llm_client.generate(narrative_prompt, temperature=0.4)
        
        # Dodaj odniesienia do wizualizacji
        for i, viz in enumerate(visualizations):
            narrative = narrative.replace(
                f"[Wykres {i+1}]",
                f"(Zobacz: {viz['title']})"
            )
            
        return narrative
```

## 3. Personalizacja treści generowanych przez modele językowe

### 3.1 System personalizacji contentu

```python
class ContentPersonalizationSystem:
    def __init__(self):
        self.user_profiler = UserProfiler()
        self.content_adapter = ContentAdapter()
        self.ab_testing_engine = ABTestingEngine()
        
    async def generate_personalized_content(self, user_id: str, 
                                          content_request: Dict) -> Dict:
        """Generuje spersonalizowaną treść dla użytkownika"""
        # 1. Pobierz profil użytkownika
        user_profile = await self.user_profiler.get_comprehensive_profile(user_id)
        
        # 2. Określ strategię personalizacji
        personalization_strategy = self.determine_strategy(user_profile, content_request)
        
        # 3. Generuj warianty treści
        content_variants = await self.generate_content_variants(
            content_request,
            user_profile,
            personalization_strategy
        )
        
        # 4. Wybierz najlepszy wariant
        selected_content = await self.select_best_variant(
            content_variants,
            user_profile,
            self.ab_testing_engine.get_test_allocation(user_id)
        )
        
        # 5. Dostosuj ostateczną wersję
        final_content = await self.content_adapter.adapt_content(
            selected_content,
            user_profile
        )
        
        # 6. Track performance
        await self.track_content_performance(user_id, final_content)
        
        return final_content
    
    def determine_strategy(self, user_profile: Dict, content_request: Dict) -> Dict:
        """Określa strategię personalizacji"""
        strategy = {
            'tone': self.determine_tone(user_profile),
            'complexity': self.determine_complexity(user_profile),
            'length': self.determine_length(user_profile),
            'style': self.determine_style(user_profile),
            'interests_weight': self.calculate_interests_weight(user_profile)
        }
        
        # Dostosuj do typu contentu
        if content_request['type'] == 'email':
            strategy['format'] = 'email'
            strategy['cta_style'] = user_profile.get('preferred_cta_style', 'standard')
        elif content_request['type'] == 'article':
            strategy['format'] = 'article'
            strategy['include_examples'] = user_profile.get('likes_examples', True)
            
        return strategy
    
    async def generate_content_variants(self, request: Dict, profile: Dict, 
                                       strategy: Dict) -> List[Dict]:
        """Generuje różne warianty treści"""
        variants = []
        
        # Wariant bazowy
        base_prompt = self.create_base_prompt(request, profile, strategy)
        base_content = await self.llm_client.generate(base_prompt, temperature=0.3)
        variants.append({
            'variant': 'base',
            'content': base_content,
            'strategy': strategy
        })
        
        # Warianty eksperymentalne
        experimental_strategies = self.generate_experimental_strategies(strategy)
        
        for exp_strategy in experimental_strategies:
            exp_prompt = self.create_base_prompt(request, profile, exp_strategy)
            exp_content = await self.llm_client.generate(exp_prompt, temperature=0.4)
            variants.append({
                'variant': f'experimental_{exp_strategy["name"]}',
                'content': exp_content,
                'strategy': exp_strategy
            })
            
        return variants
    
    def create_base_prompt(self, request: Dict, profile: Dict, strategy: Dict) -> str:
        """Tworzy prompt do generowania spersonalizowanej treści"""
        prompt = f"""
        Wygeneruj {request['type']} dla użytkownika o następującym profilu:
        
        PROFIL UŻYTKOWNIKA:
        - Poziom wiedzy: {profile['knowledge_level']}
        - Preferowany ton: {strategy['tone']}
        - Zainteresowania: {', '.join(profile['interests'][:5])}
        - Preferowana długość: {strategy['length']}
        - Styl komunikacji: {strategy['style']}
        
        TEMAT: {request['topic']}
        
        WYMAGANIA:
        - Dostosuj język do poziomu wiedzy
        - Użyj przykładów związanych z zainteresowaniami
        - Zachowaj preferowany ton i styl
        - Długość: {strategy['length']} słów
        
        DODATKOWE WSKAZÓWKI:
        {self.get_additional_guidelines(profile, strategy)}
        """
        
        return prompt
    
    async def track_content_performance(self, user_id: str, content: Dict):
        """Śledzi wydajność spersonalizowanego contentu"""
        tracking_data = {
            'user_id': user_id,
            'content_id': content['id'],
            'variant': content['variant'],
            'timestamp': datetime.now(),
            'initial_metrics': {
                'predicted_engagement': self.predict_engagement(content),
                'personalization_score': self.calculate_personalization_score(content)
            }
        }
        
        # Zapisz do systemu trackingu
        await self.save_tracking_data(tracking_data)
        
        # Zaplanuj follow-up metrics collection
        await self.schedule_performance_check(content['id'], delays=[1, 24, 168])  # 1h, 1d, 1w
```

### 3.2 Dynamiczne dostosowanie treści

```python
class DynamicContentAdapter:
    def __init__(self):
        self.real_time_signals = RealTimeSignalProcessor()
        self.content_modifier = ContentModifier()
        
    async def adapt_content_in_real_time(self, base_content: str, 
                                        user_context: Dict) -> str:
        """Dostosowuje treść w czasie rzeczywistym"""
        # 1. Zbierz sygnały real-time
        signals = await self.real_time_signals.collect_signals(user_context)
        
        # 2. Określ modyfikacje
        modifications = self.determine_modifications(signals)
        
        # 3. Zastosuj modyfikacje
        adapted_content = base_content
        for modification in modifications:
            adapted_content = await self.apply_modification(
                adapted_content,
                modification,
                user_context
            )
            
        return adapted_content
    
    def determine_modifications(self, signals: Dict) -> List[Dict]:
        """Określa jakie modyfikacje zastosować"""
        modifications = []
        
        # Czas dnia
        hour = signals['timestamp'].hour
        if hour < 9:
            modifications.append({
                'type': 'time_based',
                'adjustment': 'morning_friendly'
            })
        elif hour > 20:
            modifications.append({
                'type': 'time_based',
                'adjustment': 'evening_relaxed'
            })
        
        # Urządzenie
        if signals['device'] == 'mobile':
            modifications.append({
                'type': 'device_based',
                'adjustment': 'mobile_optimized'
            })
        
        # Pogoda (jeśli dostępna)
        if 'weather' in signals:
            if signals['weather']['condition'] == 'rainy':
                modifications.append({
                    'type': 'weather_based',
                    'adjustment': 'rainy_day_mood'
                })
        
        # Aktywność użytkownika
        if signals.get('user_activity', {}).get('is_busy', False):
            modifications.append({
                'type': 'activity_based',
                'adjustment': 'concise_version'
            })
            
        return modifications
    
    async def apply_modification(self, content: str, modification: Dict, 
                                context: Dict) -> str:
        """Aplikuje pojedynczą modyfikację do treści"""
        if modification['type'] == 'time_based':
            if modification['adjustment'] == 'morning_friendly':
                prompt = f"""
                Dostosuj poniższą treść do porannej pory:
                - Dodaj przyjazne powitanie
                - Użyj energicznego, motywującego tonu
                - Możesz wspomnieć o kawie/śniadaniu jeśli pasuje
                
                Oryginalna treść: {content}
                """
            elif modification['adjustment'] == 'evening_relaxed':
                prompt = f"""
                Dostosuj poniższą treść do wieczornej pory:
                - Użyj spokojniejszego tonu
                - Unikaj zbyt energicznych call-to-action
                - Możesz dodać życzenia miłego wieczoru
                
                Oryginalna treść: {content}
                """
                
        elif modification['type'] == 'device_based':
            if modification['adjustment'] == 'mobile_optimized':
                prompt = f"""
                Optymalizuj treść dla urządzenia mobilnego:
                - Skróć długie zdania
                - Użyj krótszych akapitów
                - Wyeksponuj najważniejsze informacje na początku
                - Usuń zbędne szczegóły
                
                Oryginalna treść: {content}
                """
                
        elif modification['type'] == 'activity_based':
            if modification['adjustment'] == 'concise_version':
                prompt = f"""
                Stwórz zwięzłą wersję treści dla zabieganego użytkownika:
                - Zachowaj tylko kluczowe informacje
                - Użyj punktorów gdzie to możliwe
                - Maksymalnie 50% oryginalnej długości
                
                Oryginalna treść: {content}
                """
                
        modified_content = await self.llm_client.generate(prompt, temperature=0.2)
        return modified_content
```

### 3.3 System rekomendacji oparty na LLM

```python
class LLMRecommendationEngine:
    def __init__(self):
        self.user_history_analyzer = UserHistoryAnalyzer()
        self.content_embedder = ContentEmbedder()
        self.recommendation_generator = RecommendationGenerator()
        
    async def generate_recommendations(self, user_id: str, 
                                     recommendation_type: str,
                                     count: int = 5) -> List[Dict]:
        """Generuje spersonalizowane rekomendacje"""
        # 1. Analiza historii użytkownika
        user_history = await self.user_history_analyzer.analyze(user_id)
        
        # 2. Określ preferencje
        preferences = await self.extract_preferences(user_history)
        
        # 3. Generuj rekomendacje
        recommendations = await self.generate_llm_recommendations(
            preferences,
            recommendation_type,
            count
        )
        
        # 4. Rankuj i filtruj
        ranked_recommendations = await self.rank_recommendations(
            recommendations,
            user_history,
            preferences
        )
        
        # 5. Dodaj wyjaśnienia
        explained_recommendations = await self.add_explanations(
            ranked_recommendations,
            preferences
        )
        
        return explained_recommendations[:count]
    
    async def extract_preferences(self, user_history: Dict) -> Dict:
        """Ekstraktuje preferencje użytkownika z historii"""
        preference_prompt = f"""
        Przeanalizuj historię aktywności użytkownika i wyodrębnij preferencje:
        
        Historia zakupów: {user_history.get('purchases', [])}
        Historia przeglądania: {user_history.get('browsing', [])}
        Oceny i recenzje: {user_history.get('reviews', [])}
        Interakcje: {user_history.get('interactions', [])}
        
        Wyodrębnij:
        1. Kategorie produktów/treści preferowane przez użytkownika
        2. Cechy cenione przez użytkownika (np. jakość, cena, innowacyjność)
        3. Marki lub autorzy preferowani
        4. Przedział cenowy
        5. Styl i ton preferowanej komunikacji
        6. Negatywne preferencje (czego unika)
        
        Format: JSON
        """
        
        preferences_json = await self.llm_client.generate(
            preference_prompt,
            temperature=0.2
        )
        
        return json.loads(preferences_json)
    
    async def generate_llm_recommendations(self, preferences: Dict, 
                                         rec_type: str, count: int) -> List[Dict]:
        """Generuje rekomendacje używając LLM"""
        if rec_type == 'products':
            prompt = f"""
            Wygeneruj {count * 2} rekomendacji produktów dla użytkownika o preferencjach:
            {json.dumps(preferences, indent=2)}
            
            Dla każdej rekomendacji podaj:
            - Nazwa produktu
            - Kategoria
            - Dlaczego pasuje do użytkownika (krótko)
            - Przewidywane dopasowanie (1-10)
            - Cechy kluczowe
            - Przedział cenowy
            """
        elif rec_type == 'content':
            prompt = f"""
            Wygeneruj {count * 2} rekomendacji treści/artykułów dla użytkownika o preferencjach:
            {json.dumps(preferences, indent=2)}
            
            Dla każdej rekomendacji podaj:
            - Tytuł
            - Kategoria/temat
            - Dlaczego zainteresuje użytkownika
            - Poziom zaawansowania
            - Szacowany czas czytania
            - Powiązane tematy
            """
            
        recommendations_json = await self.llm_client.generate(
            prompt,
            temperature=0.5,
            max_tokens=2000
        )
        
        return json.loads(recommendations_json)
    
    async def add_explanations(self, recommendations: List[Dict], 
                              preferences: Dict) -> List[Dict]:
        """Dodaje spersonalizowane wyjaśnienia do rekomendacji"""
        explained_recs = []
        
        for rec in recommendations:
            explanation_prompt = f"""
            Stwórz krótkie, spersonalizowane wyjaśnienie dlaczego rekomendujesz:
            {rec['name']}
            
            Użytkownik lubi: {preferences['positive_traits']}
            Ostatnie zakupy/zainteresowania: {preferences['recent_interests']}
            
            Wyjaśnienie powinno:
            - Być osobiste ("Ponieważ lubisz X, Y może Ci się spodobać")
            - Wspomnieć konkretną cechę produktu/treści
            - Być krótkie (1-2 zdania)
            - Brzmieć naturalnie
            """
            
            explanation = await self.llm_client.generate(
                explanation_prompt,
                temperature=0.6
            )
            
            rec['personalized_explanation'] = explanation
            explained_recs.append(rec)
            
        return explained_recs
```

## 4. Case Studies i najlepsze praktyki

### 4.1 Case Study: E-commerce Chatbot

```python
class EcommerceChatbotImplementation:
    """
    Case Study: Implementacja chatbota dla sklepu e-commerce
    Cel: Zwiększenie konwersji o 25% i redukcja obciążenia support o 40%
    """
    
    def __init__(self):
        self.performance_metrics = {
            'conversion_rate': 0,
            'support_tickets_reduced': 0,
            'customer_satisfaction': 0,
            'average_resolution_time': 0
        }
        
    async def implement_full_solution(self):
        """Pełna implementacja rozwiązania"""
        # 1. Faza analizy
        requirements = await self.analyze_business_requirements()
        
        # 2. Faza projektowania
        architecture = self.design_system_architecture(requirements)
        
        # 3. Faza implementacji
        chatbot = await self.implement_chatbot(architecture)
        
        # 4. Faza integracji
        integrated_system = await self.integrate_with_existing_systems(chatbot)
        
        # 5. Faza testowania
        test_results = await self.comprehensive_testing(integrated_system)
        
        # 6. Faza wdrożenia
        deployment = await self.phased_deployment(integrated_system)
        
        # 7. Faza monitorowania
        monitoring = self.setup_monitoring(deployment)
        
        return {
            'system': integrated_system,
            'metrics': monitoring,
            'documentation': self.generate_documentation()
        }
    
    def design_conversation_flows(self) -> Dict:
        """Projektuje przepływy konwersacji"""
        flows = {
            'product_inquiry': {
                'triggers': ['szukam', 'czy macie', 'informacje o produkcie'],
                'flow': [
                    {'action': 'identify_product', 'method': 'nlp_extraction'},
                    {'action': 'fetch_product_info', 'source': 'database'},
                    {'action': 'present_info', 'format': 'card'},
                    {'action': 'suggest_similar', 'count': 3},
                    {'action': 'offer_human_help', 'condition': 'if_not_satisfied'}
                ]
            },
            'order_tracking': {
                'triggers': ['gdzie jest zamówienie', 'status przesyłki', 'tracking'],
                'flow': [
                    {'action': 'verify_customer', 'method': 'order_number_or_email'},
                    {'action': 'fetch_order_status', 'source': 'logistics_api'},
                    {'action': 'present_timeline', 'format': 'visual'},
                    {'action': 'proactive_updates', 'method': 'subscribe'}
                ]
            },
            'return_process': {
                'triggers': ['zwrot', 'reklamacja', 'nie pasuje'],
                'flow': [
                    {'action': 'empathy_statement', 'tone': 'understanding'},
                    {'action': 'identify_order', 'method': 'guided_search'},
                    {'action': 'check_return_eligibility', 'rules': 'business_rules'},
                    {'action': 'guide_return_process', 'format': 'step_by_step'},
                    {'action': 'generate_return_label', 'integration': 'shipping_api'},
                    {'action': 'confirm_and_followup', 'channel': 'email'}
                ]
            }
        }
        
        return flows
    
    def implement_proactive_features(self):
        """Implementuje proaktywne funkcje chatbota"""
        features = {
            'cart_abandonment_recovery': {
                'trigger': 'cart_inactive_15min',
                'action': self.send_personalized_reminder,
                'success_rate': 0.23  # 23% recovery rate
            },
            'size_recommendation': {
                'trigger': 'viewing_clothing',
                'action': self.offer_size_guide,
                'success_rate': 0.45  # 45% use the guide
            },
            'cross_sell_suggestions': {
                'trigger': 'add_to_cart',
                'action': self.suggest_complementary_products,
                'success_rate': 0.18  # 18% attach rate
            },
            'shipping_updates': {
                'trigger': 'order_status_change',
                'action': self.send_proactive_update,
                'satisfaction_increase': 0.15  # 15% increase
            }
        }
        
        return features
```

### 4.2 Mierzenie ROI i optymalizacja

```python
class ChatbotROICalculator:
    def __init__(self):
        self.cost_factors = {
            'development': 0,
            'infrastructure': 0,
            'maintenance': 0,
            'training': 0
        }
        self.benefit_factors = {
            'support_cost_reduction': 0,
            'increased_conversions': 0,
            'customer_retention': 0,
            'operational_efficiency': 0
        }
        
    def calculate_roi(self, implementation_data: Dict) -> Dict:
        """Kalkuluje ROI implementacji chatbota"""
        # Koszty
        total_costs = sum([
            implementation_data['development_cost'],
            implementation_data['monthly_infrastructure'] * 12,
            implementation_data['annual_maintenance'],
            implementation_data['training_cost']
        ])
        
        # Korzyści
        support_savings = (
            implementation_data['support_tickets_before'] - 
            implementation_data['support_tickets_after']
        ) * implementation_data['cost_per_ticket']
        
        conversion_increase = (
            implementation_data['conversion_rate_after'] - 
            implementation_data['conversion_rate_before']
        ) * implementation_data['annual_revenue']
        
        retention_value = (
            implementation_data['retention_rate_increase'] * 
            implementation_data['customer_lifetime_value'] * 
            implementation_data['customer_base']
        )
        
        total_benefits = support_savings + conversion_increase + retention_value
        
        roi = ((total_benefits - total_costs) / total_costs) * 100
        payback_period = total_costs / (total_benefits / 12)  # w miesiącach
        
        return {
            'roi_percentage': roi,
            'payback_period_months': payback_period,
            'annual_savings': total_benefits,
            'total_investment': total_costs,
            'breakdown': {
                'support_savings': support_savings,
                'conversion_increase': conversion_increase,
                'retention_value': retention_value
            }
        }
    
    def optimize_performance(self, current_metrics: Dict) -> List[Dict]:
        """Generuje rekomendacje optymalizacji"""
        recommendations = []
        
        # Analiza wskaźników
        if current_metrics['resolution_rate'] < 0.7:
            recommendations.append({
                'area': 'Knowledge Base',
                'issue': 'Low resolution rate',
                'recommendation': 'Expand knowledge base with top unresolved queries',
                'expected_impact': '+15% resolution rate',
                'implementation_effort': 'Medium'
            })
            
        if current_metrics['average_conversation_length'] > 10:
            recommendations.append({
                'area': 'Conversation Design',
                'issue': 'Long conversations',
                'recommendation': 'Implement better intent recognition and quick actions',
                'expected_impact': '-30% conversation length',
                'implementation_effort': 'High'
            })
            
        if current_metrics['escalation_rate'] > 0.2:
            recommendations.append({
                'area': 'Automation Scope',
                'issue': 'High escalation rate',
                'recommendation': 'Analyze escalation patterns and automate top reasons',
                'expected_impact': '-40% escalations',
                'implementation_effort': 'Medium'
            })
            
        return recommendations
```

## 5. Ćwiczenia praktyczne

### Ćwiczenie 1: Zaprojektuj chatbota
1. Wybierz branżę (np. bankowość, travel, healthcare)
2. Zidentyfikuj 5 głównych przypadków użycia
3. Zaprojektuj flow dla każdego przypadku
4. Określ punkty integracji z systemami
5. Zaplanuj metryki sukcesu

### Ćwiczenie 2: Personalizacja treści
1. Stwórz 3 persony użytkowników
2. Wygeneruj ten sam content dla każdej persony
3. Porównaj różnice w:
   - Tonie
   - Długości
   - Użytym słownictwie
   - Call-to-action
4. Zmierz które wersje są najskuteczniejsze

### Ćwiczenie 3: Generator raportów
1. Przygotuj przykładowe dane sprzedażowe
2. Zaimplementuj generator monthly report
3. Dodaj:
   - Automatyczną analizę trendów
   - Generowanie insights
   - Wizualizacje
   - Rekomendacje
4. Wygeneruj raport i oceń jakość

### Ćwiczenie 4: ROI Calculator
1. Dla wybranego use case oblicz:
   - Koszty implementacji
   - Oczekiwane oszczędności
   - Dodatkowe przychody
   - Okres zwrotu
2. Zidentyfikuj największe ryzyka
3. Zaproponuj plan mitygacji

## 6. Podsumowanie modułu

### Kluczowe wnioski:
1. **Sukces = Technologia + Proces + Ludzie**
2. **Personalizacja zwiększa engagement o 50-70%**
3. **Automatyzacja to nie zastąpienie, a wsparcie ludzi**
4. **Continuous improvement jest kluczowe**
5. **ROI jest mierzalne i znaczące**

### Checklist wdrożenia:
- [ ] Zdefiniowane cele biznesowe
- [ ] Zmapowane procesy i integracje
- [ ] Przygotowane dane treningowe
- [ ] Zaprojektowane flows konwersacji
- [ ] Zaimplementowane zabezpieczenia
- [ ] Skonfigurowany monitoring
- [ ] Przeszkolony zespół
- [ ] Plan rozwoju i optymalizacji

## Materiały dodatkowe

### Case Studies:
- "How Klarna's AI assistant handled 2.3M conversations" (2024)
- "Notion AI: Personalization at Scale" (2023)
- "Bloomberg's Automated Report Generation" (2023)

### Narzędzia:
- **Rasa** - Open source conversational AI
- **Botpress** - Visual chatbot builder
- **LangChain** - LLM application framework
- **Streamlit** - Rapid prototyping for AI apps

### Książki:
- "Conversational Design" - Erika Hall
- "Designing Bots" - Amir Shevat
- "The AI-First Company" - Ash Fontana