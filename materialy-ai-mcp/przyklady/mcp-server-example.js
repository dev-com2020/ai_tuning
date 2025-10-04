#!/usr/bin/env node
/**
 * Przykładowy serwer MCP (Model Context Protocol)
 * Demonstracja różnych funkcjonalności
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';

const execAsync = promisify(exec);

// Inicjalizacja serwera
const server = new Server({
  name: 'example-mcp-server',
  version: '1.0.0',
  description: 'Przykładowy serwer MCP z różnymi funkcjonalnościami'
});

// === ZASOBY (Resources) ===
// Udostępnianie plików i katalogów

server.setRequestHandler('resources/list', async () => {
  return {
    resources: [
      {
        uri: 'file:///workspace/data',
        name: 'Workspace Data',
        description: 'Katalog z danymi projektu',
        mimeType: 'text/directory'
      },
      {
        uri: 'file:///workspace/config.json',
        name: 'Configuration',
        description: 'Plik konfiguracyjny aplikacji',
        mimeType: 'application/json'
      },
      {
        uri: 'dynamic://system-info',
        name: 'System Information',
        description: 'Dynamiczne informacje o systemie',
        mimeType: 'application/json'
      }
    ]
  };
});

server.setRequestHandler('resources/read', async (request) => {
  const uri = request.params.uri;
  
  if (uri.startsWith('file://')) {
    // Odczyt pliku
    const filePath = uri.replace('file://', '');
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return {
        contents: [{
          uri,
          mimeType: 'text/plain',
          text: content
        }]
      };
    } catch (error) {
      throw new Error(`Nie mogę odczytać pliku: ${error.message}`);
    }
  } else if (uri === 'dynamic://system-info') {
    // Dynamiczne informacje
    const systemInfo = {
      platform: os.platform(),
      arch: os.arch(),
      cpus: os.cpus().length,
      totalMemory: `${(os.totalmem() / 1024 / 1024 / 1024).toFixed(2)} GB`,
      freeMemory: `${(os.freemem() / 1024 / 1024 / 1024).toFixed(2)} GB`,
      nodeVersion: process.version,
      uptime: `${(os.uptime() / 3600).toFixed(2)} hours`
    };
    
    return {
      contents: [{
        uri,
        mimeType: 'application/json',
        text: JSON.stringify(systemInfo, null, 2)
      }]
    };
  }
  
  throw new Error(`Nieznany zasób: ${uri}`);
});

// === NARZĘDZIA (Tools) ===
// Funkcjonalności dostępne dla klienta

// 1. Narzędzie do analizy plików
const analyzeFileSchema = z.object({
  path: z.string().describe('Ścieżka do pliku'),
  detailed: z.boolean().optional().describe('Czy zwrócić szczegółową analizę')
});

server.setRequestHandler('tools/list', async () => {
  return {
    tools: [
      {
        name: 'analyze_file',
        description: 'Analizuje plik i zwraca statystyki',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Ścieżka do pliku' },
            detailed: { type: 'boolean', description: 'Szczegółowa analiza' }
          },
          required: ['path']
        }
      },
      {
        name: 'execute_command',
        description: 'Wykonuje polecenie systemowe',
        inputSchema: {
          type: 'object',
          properties: {
            command: { type: 'string', description: 'Polecenie do wykonania' },
            cwd: { type: 'string', description: 'Katalog roboczy' }
          },
          required: ['command']
        }
      },
      {
        name: 'search_files',
        description: 'Wyszukuje pliki według wzorca',
        inputSchema: {
          type: 'object',
          properties: {
            pattern: { type: 'string', description: 'Wzorzec wyszukiwania (regex)' },
            directory: { type: 'string', description: 'Katalog do przeszukania' },
            maxResults: { type: 'number', description: 'Maksymalna liczba wyników' }
          },
          required: ['pattern']
        }
      },
      {
        name: 'create_project',
        description: 'Tworzy strukturę nowego projektu',
        inputSchema: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'Nazwa projektu' },
            type: { 
              type: 'string', 
              enum: ['web', 'api', 'ml', 'cli'],
              description: 'Typ projektu' 
            },
            language: {
              type: 'string',
              enum: ['javascript', 'python', 'typescript'],
              description: 'Język programowania'
            }
          },
          required: ['name', 'type', 'language']
        }
      }
    ]
  };
});

// Obsługa wywołań narzędzi
server.setRequestHandler('tools/call', async (request) => {
  const { name, arguments: args } = request.params;
  
  switch (name) {
    case 'analyze_file':
      return await analyzeFile(args);
      
    case 'execute_command':
      return await executeCommand(args);
      
    case 'search_files':
      return await searchFiles(args);
      
    case 'create_project':
      return await createProject(args);
      
    default:
      throw new Error(`Nieznane narzędzie: ${name}`);
  }
});

// Implementacje narzędzi

async function analyzeFile(args) {
  const { path: filePath, detailed = false } = args;
  
  try {
    const stats = await fs.stat(filePath);
    const content = await fs.readFile(filePath, 'utf-8');
    
    const analysis = {
      path: filePath,
      size: `${(stats.size / 1024).toFixed(2)} KB`,
      lines: content.split('\n').length,
      characters: content.length,
      lastModified: stats.mtime.toISOString()
    };
    
    if (detailed) {
      // Dodatkowa analiza
      const extension = path.extname(filePath);
      
      if (['.js', '.ts', '.py'].includes(extension)) {
        // Analiza kodu
        analysis.codeStats = {
          functions: (content.match(/function\s+\w+/g) || []).length,
          classes: (content.match(/class\s+\w+/g) || []).length,
          imports: (content.match(/import\s+.+from/g) || []).length,
          comments: (content.match(/\/\/.*|\/\*[\s\S]*?\*\//g) || []).length
        };
      }
      
      if (['.json'].includes(extension)) {
        try {
          const jsonData = JSON.parse(content);
          analysis.jsonStats = {
            keys: Object.keys(jsonData).length,
            type: Array.isArray(jsonData) ? 'array' : 'object'
          };
        } catch (e) {
          analysis.jsonStats = { error: 'Invalid JSON' };
        }
      }
    }
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(analysis, null, 2)
      }]
    };
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: `Błąd analizy pliku: ${error.message}`
      }]
    };
  }
}

async function executeCommand(args) {
  const { command, cwd = process.cwd() } = args;
  
  // Lista bezpiecznych komend
  const safeCommands = ['ls', 'pwd', 'echo', 'date', 'node --version', 'python --version'];
  const isCommandSafe = safeCommands.some(safe => command.startsWith(safe));
  
  if (!isCommandSafe) {
    return {
      content: [{
        type: 'text',
        text: `⚠️ Komenda '${command}' nie jest na liście bezpiecznych komend.`
      }]
    };
  }
  
  try {
    const { stdout, stderr } = await execAsync(command, { cwd });
    
    return {
      content: [{
        type: 'text',
        text: `Wykonano: ${command}\n\nOutput:\n${stdout}\n${stderr ? `\nErrors:\n${stderr}` : ''}`
      }]
    };
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: `Błąd wykonania komendy: ${error.message}`
      }]
    };
  }
}

async function searchFiles(args) {
  const { pattern, directory = '.', maxResults = 10 } = args;
  const results = [];
  
  async function searchDir(dir) {
    if (results.length >= maxResults) return;
    
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        if (results.length >= maxResults) break;
        
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.')) {
          await searchDir(fullPath);
        } else if (entry.isFile()) {
          const regex = new RegExp(pattern, 'i');
          
          // Szukaj w nazwie pliku
          if (regex.test(entry.name)) {
            results.push({
              path: fullPath,
              type: 'filename_match'
            });
          }
          
          // Szukaj w zawartości (tylko pliki tekstowe)
          if (results.length < maxResults && 
              ['.txt', '.js', '.py', '.md', '.json'].includes(path.extname(entry.name))) {
            try {
              const content = await fs.readFile(fullPath, 'utf-8');
              if (regex.test(content)) {
                const matches = content.match(new RegExp(`.{0,50}${pattern}.{0,50}`, 'gi'));
                results.push({
                  path: fullPath,
                  type: 'content_match',
                  preview: matches ? matches[0] : ''
                });
              }
            } catch (e) {
              // Ignoruj błędy odczytu
            }
          }
        }
      }
    } catch (error) {
      console.error(`Błąd przeszukiwania ${dir}: ${error.message}`);
    }
  }
  
  await searchDir(directory);
  
  return {
    content: [{
      type: 'text',
      text: `Znaleziono ${results.length} wyników dla wzorca "${pattern}":\n\n` +
            results.map(r => `${r.path} (${r.type})${r.preview ? `\n  Preview: ${r.preview}` : ''}`).join('\n\n')
    }]
  };
}

async function createProject(args) {
  const { name, type, language } = args;
  const projectPath = path.join(process.cwd(), name);
  
  // Struktury projektów
  const templates = {
    web: {
      javascript: {
        'index.html': `<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${name}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>Witaj w ${name}</h1>
    <script src="script.js"></script>
</body>
</html>`,
        'script.js': `// ${name} - Main JavaScript file
console.log('Hello from ${name}!');`,
        'styles.css': `/* ${name} - Styles */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}`,
        'package.json': JSON.stringify({
          name: name.toLowerCase().replace(/\s+/g, '-'),
          version: '1.0.0',
          description: `${name} web project`,
          main: 'script.js',
          scripts: {
            start: 'open index.html'
          }
        }, null, 2)
      },
      python: {
        'app.py': `# ${name} - Flask Web Application
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', project_name='${name}')

if __name__ == '__main__':
    app.run(debug=True)`,
        'templates/index.html': `<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>{{ project_name }}</title>
</head>
<body>
    <h1>Witaj w {{ project_name }}</h1>
</body>
</html>`,
        'requirements.txt': 'flask==2.3.2\n',
        'README.md': `# ${name}

Web application built with Flask.

## Setup
\`\`\`bash
pip install -r requirements.txt
python app.py
\`\`\`
`
      }
    },
    api: {
      javascript: {
        'index.js': `// ${name} - Express API Server
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

app.get('/', (req, res) => {
    res.json({ message: 'Welcome to ${name} API' });
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date() });
});

app.listen(PORT, () => {
    console.log(\`Server running on port \${PORT}\`);
});`,
        'package.json': JSON.stringify({
          name: name.toLowerCase().replace(/\s+/g, '-'),
          version: '1.0.0',
          description: `${name} API server`,
          main: 'index.js',
          scripts: {
            start: 'node index.js',
            dev: 'nodemon index.js'
          },
          dependencies: {
            express: '^4.18.2'
          },
          devDependencies: {
            nodemon: '^3.0.1'
          }
        }, null, 2),
        '.env.example': 'PORT=3000\nNODE_ENV=development\n',
        '.gitignore': 'node_modules/\n.env\n'
      },
      python: {
        'main.py': `# ${name} - FastAPI Application
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="${name}")

@app.get("/")
def root():
    return {"message": "Welcome to ${name} API"}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }`,
        'requirements.txt': 'fastapi==0.104.1\nuvicorn==0.24.0\n',
        'run.sh': `#!/bin/bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000`
      }
    },
    ml: {
      python: {
        'train.py': `# ${name} - Machine Learning Training Script
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    """Load and prepare data"""
    # TODO: Implement data loading
    pass

def train_model(X_train, y_train):
    """Train the model"""
    # TODO: Implement model training
    pass

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # TODO: Implement evaluation
    pass

if __name__ == "__main__":
    print("Starting ${name} training...")
    # Load data
    # Train model
    # Save model
    print("Training completed!")`,
        'predict.py': `# ${name} - Prediction Script
import joblib
import numpy as np

def load_model(path='model.pkl'):
    """Load trained model"""
    return joblib.load(path)

def predict(model, data):
    """Make predictions"""
    return model.predict(data)

if __name__ == "__main__":
    # Example usage
    model = load_model()
    # Make predictions`,
        'requirements.txt': `numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.1
joblib==1.3.1
jupyter==1.0.0`,
        'notebooks/.gitkeep': '',
        'data/.gitkeep': '',
        'models/.gitkeep': ''
      }
    }
  };
  
  // Sprawdź czy projekt już istnieje
  try {
    await fs.access(projectPath);
    return {
      content: [{
        type: 'text',
        text: `❌ Projekt '${name}' już istnieje w tej lokalizacji.`
      }]
    };
  } catch {
    // Projekt nie istnieje, możemy kontynuować
  }
  
  // Pobierz szablon
  const template = templates[type]?.[language];
  if (!template) {
    return {
      content: [{
        type: 'text',
        text: `❌ Brak szablonu dla typu '${type}' i języka '${language}'.`
      }]
    };
  }
  
  // Utwórz strukturę projektu
  try {
    await fs.mkdir(projectPath, { recursive: true });
    
    for (const [filePath, content] of Object.entries(template)) {
      const fullPath = path.join(projectPath, filePath);
      const dir = path.dirname(fullPath);
      
      await fs.mkdir(dir, { recursive: true });
      await fs.writeFile(fullPath, content);
    }
    
    // Dodaj wspólne pliki
    const commonFiles = {
      '.gitignore': `# ${name}
node_modules/
*.log
.env
.DS_Store
__pycache__/
*.pyc
.vscode/
.idea/`,
      'README.md': `# ${name}

${type.charAt(0).toUpperCase() + type.slice(1)} project created with MCP.

## Technology
- Language: ${language}
- Type: ${type}

## Getting Started
See the specific files for setup instructions.

## Structure
\`\`\`
${name}/
${Object.keys(template).map(f => `├── ${f}`).join('\n')}
└── README.md
\`\`\`
`
    };
    
    for (const [filePath, content] of Object.entries(commonFiles)) {
      await fs.writeFile(path.join(projectPath, filePath), content);
    }
    
    return {
      content: [{
        type: 'text',
        text: `✅ Projekt '${name}' został utworzony!\n\n` +
              `Typ: ${type}\n` +
              `Język: ${language}\n` +
              `Lokalizacja: ${projectPath}\n\n` +
              `Pliki:\n${Object.keys(template).map(f => `- ${f}`).join('\n')}`
      }]
    };
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: `❌ Błąd tworzenia projektu: ${error.message}`
      }]
    };
  }
}

// === PROMPTS ===
// Sugestie promptów dla użytkownika

server.setRequestHandler('prompts/list', async () => {
  return {
    prompts: [
      {
        name: 'analyze_codebase',
        description: 'Analizuj strukturę i jakość kodu',
        arguments: [
          {
            name: 'path',
            description: 'Ścieżka do projektu',
            required: true
          }
        ]
      },
      {
        name: 'refactor_code',
        description: 'Zaproponuj refaktoryzację kodu',
        arguments: [
          {
            name: 'file',
            description: 'Plik do refaktoryzacji',
            required: true
          },
          {
            name: 'focus',
            description: 'Obszar do poprawy (performance, readability, etc.)',
            required: false
          }
        ]
      }
    ]
  };
});

// === COMPLETION ===
// Autouzupełnianie

server.setRequestHandler('completion/complete', async (request) => {
  const { ref, argument, values } = request.params;
  
  if (ref.name === 'create_project' && argument.name === 'name') {
    // Sugestie nazw projektów
    return {
      completion: {
        values: [
          'my-awesome-app',
          'test-project',
          'demo-api',
          'ml-experiment'
        ]
      }
    };
  }
  
  return { completion: { values: [] } };
});

// === URUCHOMIENIE SERWERA ===

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  console.error('MCP Server started successfully!');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});