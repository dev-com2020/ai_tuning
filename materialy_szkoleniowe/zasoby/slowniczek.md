# Słowniczek pojęć - LLM i AI

## A

**Attention Mechanism**
- Mechanizm pozwalający modelowi "skupić uwagę" na różnych częściach tekstu wejściowego podczas generowania odpowiedzi. Kluczowy komponent architektury Transformer.

**API (Application Programming Interface)**
- Interfejs programowania aplikacji umożliwiający komunikację z modelem językowym poprzez zapytania HTTP.

**Augmentacja danych**
- Technika zwiększania ilości danych treningowych poprzez tworzenie zmodyfikowanych wersji istniejących przykładów.

## B

**BERT (Bidirectional Encoder Representations from Transformers)**
- Model językowy Google analizujący tekst w obu kierunkach jednocześnie, używany głównie do zadań rozumienia języka.

**BERTScore**
- Metryka oceny jakości tekstu wykorzystująca embeddingi z modelu BERT do pomiaru semantycznego podobieństwa.

**Bias (Tendencyjność)**
- Systematyczne błędy lub uprzedzenia w odpowiedziach modelu wynikające z niezbalansowanych danych treningowych.

**BLEU (Bilingual Evaluation Understudy)**
- Metryka oceny jakości tłumaczeń i generowanego tekstu poprzez porównanie z tekstem referencyjnym.

## C

**Chain-of-Thought (CoT)**
- Technika promptowania zachęcająca model do pokazania procesu rozumowania krok po kroku.

**Chatbot**
- System konwersacyjny wykorzystujący LLM do prowadzenia dialogu z użytkownikami.

**Claude**
- Rodzina modeli językowych stworzonych przez Anthropic, znanych z długiego kontekstu i constitutional AI.

**Constitutional AI**
- Metoda trenowania modeli AI z wbudowanymi zasadami etycznymi i ograniczeniami.

**Context Window**
- Maksymalna liczba tokenów, które model może przetworzyć jednocześnie (np. 4k, 8k, 100k tokenów).

**Corpus**
- Zbiór tekstów używany do trenowania lub ewaluacji modelu językowego.

## D

**Decoder**
- Część architektury Transformer odpowiedzialna za generowanie tekstu wyjściowego.

**Deployment**
- Proces wdrażania modelu do środowiska produkcyjnego.

## E

**Embedding**
- Numeryczna reprezentacja tekstu w postaci wektora liczb, umożliwiająca przetwarzanie przez model.

**Encoder**
- Część architektury Transformer przetwarzająca tekst wejściowy na reprezentację wewnętrzną.

**Epoch**
- Pojedyncze przejście przez cały zbiór danych treningowych podczas uczenia modelu.

**Evaluation Metrics**
- Miary używane do oceny jakości i wydajności modelu (np. perplexity, BLEU, accuracy).

## F

**Few-shot Learning**
- Technika uczenia modelu na podstawie kilku przykładów podanych w prompcie.

**Fine-tuning**
- Proces dostosowywania wstępnie wytrenowanego modelu do konkretnego zadania lub domeny.

**Frequency Penalty**
- Parametr kontrolujący jak bardzo model unika powtarzania tych samych słów.

## G

**GPT (Generative Pre-trained Transformer)**
- Seria modeli językowych OpenAI wykorzystujących architekturę Transformer do generowania tekstu.

**Gradient Descent**
- Algorytm optymalizacji używany do trenowania sieci neuronowych.

## H

**Hallucination (Halucynacja)**
- Zjawisko generowania przez model nieprawdziwych lub zmyślonych informacji prezentowanych jako fakty.

**Human Evaluation**
- Ocena jakości odpowiedzi modelu przez ludzi, często uznawana za "złoty standard".

**Hyperparameters**
- Parametry konfiguracji modelu ustawiane przed treningiem (np. learning rate, batch size).

## I

**Inference**
- Proces generowania odpowiedzi przez wytrenowany model.

**Intent Recognition**
- Rozpoznawanie intencji użytkownika na podstawie jego zapytania.

**Inter-rater Agreement**
- Miara zgodności między różnymi osobami oceniającymi te same dane.

## J

**JSON (JavaScript Object Notation)**
- Format wymiany danych często używany w API i do strukturyzacji promptów.

**JSONL (JSON Lines)**
- Format pliku gdzie każda linia zawiera osobny obiekt JSON, używany do danych treningowych.

## K

**Knowledge Base**
- Baza wiedzy używana do wzbogacania odpowiedzi modelu o faktyczne informacje.

**k-shot Learning**
- Ogólny termin dla technik uczenia z k przykładami (gdzie k może być 0, 1, few, itp.).

## L

**Large Language Model (LLM)**
- Duży model językowy trenowany na ogromnych zbiorach tekstów, zdolny do rozumienia i generowania języka naturalnego.

**Latency**
- Czas odpowiedzi systemu, od wysłania zapytania do otrzymania odpowiedzi.

**Learning Rate**
- Hiperparametr określający jak szybko model uczy się podczas treningu.

**Loss Function**
- Funkcja mierząca różnicę między przewidywaniami modelu a oczekiwanymi wynikami.

## M

**Max Tokens**
- Maksymalna liczba tokenów w generowanej odpowiedzi.

**Model Card**
- Dokumentacja opisująca charakterystykę, ograniczenia i odpowiednie zastosowania modelu.

**Multi-head Attention**
- Mechanizm pozwalający modelowi zwracać uwagę na różne aspekty tekstu równolegle.

## N

**Natural Language Processing (NLP)**
- Dziedzina AI zajmująca się przetwarzaniem i rozumieniem języka naturalnego.

**Nucleus Sampling (Top-p)**
- Metoda próbkowania wybierająca tokeny z górnego percentyla rozkładu prawdopodobieństwa.

## O

**One-shot Learning**
- Technika uczenia modelu na podstawie pojedynczego przykładu w prompcie.

**OpenAI**
- Firma twórca modeli GPT, ChatGPT i DALL-E.

**Overfitting**
- Zjawisko nadmiernego dopasowania modelu do danych treningowych, skutkujące słabą generalizacją.

## P

**Parameter**
- Waga w sieci neuronowej uczona podczas treningu.

**Perplexity (Perpleksja)**
- Miara "zaskoczenia" modelu przez tekst; niższa wartość oznacza lepszy model.

**Presence Penalty**
- Parametr kontrolujący tendencję modelu do wprowadzania nowych tematów.

**Prompt**
- Instrukcja lub zapytanie przekazywane modelowi językowemu.

**Prompt Engineering**
- Sztuka tworzenia efektywnych promptów maksymalizujących jakość odpowiedzi.

## Q

**QLoRA**
- Technika efektywnego fine-tuningu wykorzystująca kwantyzację dla redukcji zużycia pamięci.

**Query**
- Zapytanie użytkownika do modelu.

## R

**RAG (Retrieval Augmented Generation)**
- Technika łącząca generowanie tekstu z wyszukiwaniem informacji z zewnętrznej bazy wiedzy.

**Reinforcement Learning from Human Feedback (RLHF)**
- Metoda uczenia modelu wykorzystująca feedback od ludzi do poprawy jakości odpowiedzi.

**ROUGE**
- Metryka oceny jakości streszczeń poprzez porównanie z referencyjnymi streszczeniami.

## S

**Sampling**
- Proces wybierania kolejnego tokenu podczas generowania tekstu.

**Self-Attention**
- Mechanizm pozwalający modelowi analizować relacje między różnymi częściami tego samego tekstu.

**System Prompt**
- Stała instrukcja definiująca zachowanie i rolę modelu w konwersacji.

## T

**Temperature**
- Parametr kontrolujący losowość/kreatywność generowanych odpowiedzi (0 = deterministyczne, >1 = kreatywne).

**Token**
- Podstawowa jednostka tekstu przetwarzana przez model (może być słowem, częścią słowa lub znakiem).

**Tokenization**
- Proces dzielenia tekstu na tokeny.

**Top-k Sampling**
- Metoda próbkowania ograniczająca wybór do k najbardziej prawdopodobnych tokenów.

**Top-p (Nucleus Sampling)**
- Metoda próbkowania wybierająca tokeny, których łączne prawdopodobieństwo nie przekracza p.

**Training Data**
- Dane używane do uczenia modelu.

**Transformer**
- Architektura sieci neuronowej będąca podstawą współczesnych LLM.

## U

**Underfitting**
- Zjawisko niedostatecznego dopasowania modelu do danych, skutkujące słabą wydajnością.

**User Prompt**
- Konkretne zapytanie lub instrukcja od użytkownika.

## V

**Validation Set**
- Zbiór danych używany do oceny modelu podczas treningu.

**Vector Database**
- Baza danych optymalizowana do przechowywania i wyszukiwania embeddingów.

## W

**Weight**
- Parametr w sieci neuronowej określający siłę połączenia między neuronami.

**Word Embedding**
- Reprezentacja słowa jako wektora liczb.

## Z

**Zero-shot Learning**
- Zdolność modelu do wykonywania zadań bez żadnych przykładów w prompcie.