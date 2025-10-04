# Wprowadzenie do AI i MCP (Model Context Protocol)

## Spis treści
1. [Czym jest AI?](#czym-jest-ai)
2. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
3. [Zastosowania AI i MCP](#zastosowania-ai-i-mcp)
4. [Architektura MCP](#architektura-mcp)
5. [Rozpoczęcie pracy](#rozpoczęcie-pracy)

## Czym jest AI?

Sztuczna inteligencja (AI) to dziedzina informatyki zajmująca się tworzeniem systemów zdolnych do wykonywania zadań wymagających ludzkiej inteligencji. Współczesne AI opiera się głównie na:

### Machine Learning (ML)
- **Uczenie nadzorowane**: Algorytmy uczą się na podstawie oznaczonych danych
- **Uczenie nienadzorowane**: Odkrywanie wzorców w nieoznaczonych danych
- **Uczenie ze wzmocnieniem**: Nauka przez interakcję ze środowiskiem

### Deep Learning
- Sieci neuronowe o wielu warstwach
- Transformery (GPT, BERT, Claude)
- Sieci konwolucyjne (CNN) dla wizji komputerowej
- Sieci rekurencyjne (RNN, LSTM) dla sekwencji

### Natural Language Processing (NLP)
- Rozumienie języka naturalnego
- Generowanie tekstu
- Tłumaczenie maszynowe
- Analiza sentymentu

## Model Context Protocol (MCP)

MCP to otwarty protokół umożliwiający bezpieczną komunikację między aplikacjami AI a lokalnymi zasobami. Został stworzony przez Anthropic jako standard integracji LLM z narzędziami zewnętrznymi.

### Kluczowe cechy MCP:
1. **Bezpieczeństwo**: Kontrolowana ekspozycja zasobów
2. **Standaryzacja**: Jednolity sposób komunikacji
3. **Modularność**: Łatwe dodawanie nowych funkcjonalności
4. **Lokalność**: Dane pozostają na urządzeniu użytkownika

### Komponenty MCP:
- **Serwery MCP**: Udostępniają zasoby i funkcjonalności
- **Klienci MCP**: Aplikacje AI korzystające z zasobów
- **Transport**: Protokół komunikacji (stdio, HTTP)
- **Zasoby**: Pliki, bazy danych, API, narzędzia

## Zastosowania AI i MCP

### AI w praktyce:
1. **Asystenci kodowania**: GitHub Copilot, Cursor, Codeium
2. **Chatboty**: ChatGPT, Claude, Gemini
3. **Generowanie obrazów**: DALL-E, Midjourney, Stable Diffusion
4. **Analiza danych**: Predykcje, klasyfikacja, klasteryzacja
5. **Automatyzacja**: RPA, procesowanie dokumentów

### MCP w ekosystemie AI:
1. **Integracja z IDE**: Dostęp do plików projektu
2. **Bazy danych**: Bezpośrednie zapytania SQL
3. **API zewnętrzne**: Pogoda, giełda, wiadomości
4. **Narzędzia systemowe**: Zarządzanie plikami, procesy
5. **Własne serwery**: Specjalistyczne funkcjonalności

## Architektura MCP

```
┌─────────────┐     MCP Protocol      ┌─────────────┐
│   Client    │ ◄─────────────────► │   Server    │
│   (AI App)  │                      │  (Resource) │
└─────────────┘                      └─────────────┘
      │                                      │
      │                                      │
      ▼                                      ▼
┌─────────────┐                      ┌─────────────┐
│    User     │                      │   Local     │
│  Interface  │                      │  Resources  │
└─────────────┘                      └─────────────┘
```

### Przepływ komunikacji:
1. Klient AI otrzymuje żądanie użytkownika
2. Identyfikuje potrzebne zasoby
3. Wysyła żądanie do serwera MCP
4. Serwer przetwarza żądanie
5. Zwraca wyniki do klienta
6. Klient generuje odpowiedź dla użytkownika

## Rozpoczęcie pracy

### Wymagania:
- **System operacyjny**: Windows 10+, macOS 10.15+, Linux
- **Node.js**: v18 lub nowszy
- **Python**: 3.8+ (dla niektórych serwerów)
- **IDE**: VS Code, Cursor lub podobne

### Pierwsze kroki:
1. Zainstaluj wymagane oprogramowanie
2. Sklonuj przykładowy serwer MCP
3. Skonfiguruj klienta AI
4. Przetestuj podstawową komunikację
5. Rozszerz funkcjonalności

### Przykład prostego serwera MCP (Node.js):

```javascript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'moj-serwer-mcp',
  version: '1.0.0',
});

// Dodaj narzędzie
server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'przywitaj',
    description: 'Przywitaj użytkownika',
    inputSchema: {
      type: 'object',
      properties: {
        imie: { type: 'string' }
      }
    }
  }]
}));

// Obsłuż wywołanie narzędzia
server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'przywitaj') {
    const imie = request.params.arguments.imie;
    return {
      content: [{
        type: 'text',
        text: `Witaj, ${imie}! Miło Cię poznać.`
      }]
    };
  }
});

// Uruchom serwer
const transport = new StdioServerTransport();
await server.connect(transport);
```

### Dalsze kroki:
- Przejdź do przewodników fine-tuningu dla [Windows](../fine-tuning/windows/przewodnik-windows.md) lub [Mac](../fine-tuning/mac/przewodnik-mac.md)
- Zobacz [przykłady kodu](../przyklady/README.md)
- Eksploruj [narzędzia pomocnicze](../narzedzia/README.md)

## Podsumowanie

AI i MCP otwierają nowe możliwości w tworzeniu inteligentnych aplikacji. MCP standaryzuje sposób, w jaki modele AI mogą bezpiecznie korzystać z lokalnych zasobów, zachowując pełną kontrolę użytkownika nad danymi. To kluczowy krok w kierunku bardziej użytecznych i zintegrowanych systemów AI.

W kolejnych rozdziałach dowiesz się, jak przeprowadzić fine-tuning modeli AI na własnych danych oraz jak stworzyć własne serwery MCP dostosowane do Twoich potrzeb.