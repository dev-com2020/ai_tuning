# Moduł 4: Fine-tuning dużych modeli językowych

## Cel modułu
Po zakończeniu tego modułu uczestnik będzie:
- Rozumiał koncepcję i zastosowania fine-tuningu
- Potrafił przygotować dane do fine-tuningu
- Znał proces dostosowywania modeli w różnych środowiskach
- Umiał ocenić kiedy fine-tuning jest właściwym rozwiązaniem

## 1. Czym jest fine-tuning i kiedy go stosować?

### 1.1 Definicja i koncepcja

**Fine-tuning** to proces dalszego trenowania wstępnie wytrenowanego modelu na specyficznym zbiorze danych, aby dostosować go do konkretnego zadania lub domeny.

```
Pre-trained Model (GPT-3, BERT)
        ↓
    Fine-tuning
        ↓
Specialized Model (Medical GPT, Legal BERT)
```

### 1.2 Pre-training vs Fine-tuning

| Aspekt | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| **Dane** | Ogromne zbiory tekstów (TB) | Małe, specyficzne zbiory (MB-GB) |
| **Czas** | Tygodnie/miesiące | Godziny/dni |
| **Koszt** | Bardzo wysoki ($$$$$) | Relatywnie niski ($-$$) |
| **Cel** | Ogólne rozumienie języka | Specjalizacja w zadaniu |
| **Wymagania** | Superkomputery | GPU konsumenckie/chmura |

### 1.3 Kiedy stosować fine-tuning?

**Fine-tuning jest wskazany gdy:**
1. **Specyficzna domena** - terminologia branżowa (medycyna, prawo, finanse)
2. **Unikalne zadanie** - format odpowiedzi niedostępny w base model
3. **Styl korporacyjny** - spójny ton marki
4. **Wydajność** - szybsze/tańsze wywołania niż długie prompty
5. **Prywatność** - dane wrażliwe pozostają w organizacji

**Fine-tuning NIE jest potrzebny gdy:**
1. **Prompt engineering wystarcza** - dobre wyniki z promptami
2. **Mało danych** - <1000 wysokiej jakości przykładów
3. **Częste zmiany** - wymagania zmieniają się dynamicznie
4. **Ogólne zadania** - model bazowy radzi sobie dobrze

### 1.4 Alternatywy do fine-tuningu

```python
# 1. Prompt Engineering
optimized_prompt = """
Jesteś ekspertem prawnym specjalizującym się w RODO.
Używaj następującego formatu odpowiedzi:
- Podstawa prawna: [artykuł]
- Interpretacja: [wyjaśnienie]
- Rekomendacja: [działanie]
"""

# 2. RAG (Retrieval Augmented Generation)
context = retrieve_relevant_docs(query)
prompt = f"Bazując na kontekście: {context}\nOdpowiedz: {query}"

# 3. Few-shot learning
examples = load_examples()
prompt = create_few_shot_prompt(examples, query)
```

## 2. Przygotowanie danych do fine-tuningu

### 2.1 Formaty danych

#### OpenAI Format (JSONL)
```json
{"messages": [{"role": "system", "content": "Jesteś asystentem prawnym."}, {"role": "user", "content": "Co to jest RODO?"}, {"role": "assistant", "content": "RODO to Rozporządzenie o Ochronie Danych Osobowych..."}]}
{"messages": [{"role": "system", "content": "Jesteś asystentem prawnym."}, {"role": "user", "content": "Jakie są kary za naruszenie RODO?"}, {"role": "assistant", "content": "Kary za naruszenie RODO mogą wynosić..."}]}
```

#### Hugging Face Format
```python
dataset = {
    "text": [
        "Pytanie: Co to jest RODO?\nOdpowiedź: RODO to Rozporządzenie...",
        "Pytanie: Jakie są kary?\nOdpowiedź: Kary mogą wynosić..."
    ],
    # lub
    "instruction": ["Co to jest RODO?", "Jakie są kary?"],
    "response": ["RODO to Rozporządzenie...", "Kary mogą wynosić..."]
}
```

### 2.2 Czyszczenie i walidacja danych

```python
class DataPreprocessor:
    def __init__(self, min_length=10, max_length=2048):
        self.min_length = min_length
        self.max_length = max_length
        self.quality_checks = []
        
    def clean_dataset(self, data):
        """Czyści i waliduje dataset"""
        cleaned_data = []
        stats = {"total": len(data), "removed": 0, "modified": 0}
        
        for item in data:
            # Podstawowe czyszczenie
            cleaned_item = self.basic_cleaning(item)
            
            # Walidacja jakości
            if self.validate_quality(cleaned_item):
                cleaned_data.append(cleaned_item)
            else:
                stats["removed"] += 1
                
        return cleaned_data, stats
    
    def basic_cleaning(self, item):
        """Podstawowe czyszczenie tekstu"""
        # Usuń nadmiarowe białe znaki
        item['instruction'] = ' '.join(item['instruction'].split())
        item['response'] = ' '.join(item['response'].split())
        
        # Usuń znaki kontrolne
        item['instruction'] = ''.join(ch for ch in item['instruction'] 
                                     if ch.isprintable() or ch.isspace())
        
        # Normalizuj kodowanie
        item['instruction'] = item['instruction'].encode('utf-8', 'ignore').decode('utf-8')
        item['response'] = item['response'].encode('utf-8', 'ignore').decode('utf-8')
        
        return item
    
    def validate_quality(self, item):
        """Sprawdza jakość przykładu"""
        # Długość
        if len(item['response']) < self.min_length:
            return False
        if len(item['response']) > self.max_length:
            return False
            
        # Kompletność
        if not item['instruction'] or not item['response']:
            return False
            
        # Język (przykład dla polskiego)
        if not self.is_polish(item['response']):
            return False
            
        # Duplikaty instrukcji w odpowiedzi
        if item['instruction'].lower() in item['response'].lower():
            return False
            
        return True
    
    def is_polish(self, text):
        """Sprawdza czy tekst jest po polsku"""
        polish_chars = set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')
        return any(char in polish_chars for char in text)
```

### 2.3 Augmentacja danych

```python
class DataAugmenter:
    def __init__(self):
        self.augmentation_techniques = {
            'paraphrase': self.paraphrase,
            'back_translation': self.back_translate,
            'template_variation': self.vary_templates,
            'noise_injection': self.add_noise
        }
    
    def paraphrase(self, text, num_variations=3):
        """Generuje parafrazy używając LLM"""
        prompt = f"""
        Sparafrazuj poniższe zdanie na {num_variations} różne sposoby:
        "{text}"
        
        Zachowaj sens, ale użyj innych słów i struktur.
        """
        # Wywołanie LLM
        variations = self.llm.generate(prompt)
        return self.parse_variations(variations)
    
    def vary_templates(self, instruction, response):
        """Tworzy wariacje używając templates"""
        templates = [
            f"{instruction}",
            f"Proszę {instruction.lower()}",
            f"Czy możesz {instruction.lower()}?",
            f"Potrzebuję pomocy z: {instruction}",
            f"Zadanie: {instruction}"
        ]
        
        return [(template, response) for template in templates]
    
    def create_balanced_dataset(self, data, target_size=1000):
        """Tworzy zbalansowany dataset"""
        # Grupuj po kategoriach
        categories = self.categorize_data(data)
        
        # Oblicz ile przykładów per kategoria
        examples_per_category = target_size // len(categories)
        
        balanced_data = []
        for category, examples in categories.items():
            if len(examples) >= examples_per_category:
                # Downsample
                balanced_data.extend(
                    random.sample(examples, examples_per_category)
                )
            else:
                # Upsample przez augmentację
                balanced_data.extend(examples)
                needed = examples_per_category - len(examples)
                augmented = self.augment_category(examples, needed)
                balanced_data.extend(augmented)
                
        return balanced_data
```

### 2.4 Walidacja datasetu

```python
class DatasetValidator:
    def __init__(self):
        self.validation_report = {}
        
    def comprehensive_validation(self, dataset):
        """Przeprowadza kompleksową walidację"""
        self.validation_report = {
            'size_analysis': self.analyze_size(dataset),
            'quality_metrics': self.check_quality(dataset),
            'diversity_score': self.measure_diversity(dataset),
            'balance_check': self.check_balance(dataset),
            'contamination': self.detect_contamination(dataset)
        }
        
        return self.validation_report
    
    def analyze_size(self, dataset):
        """Analiza wielkości datasetu"""
        return {
            'total_examples': len(dataset),
            'avg_instruction_length': np.mean([len(d['instruction']) for d in dataset]),
            'avg_response_length': np.mean([len(d['response']) for d in dataset]),
            'total_tokens': sum(self.count_tokens(d) for d in dataset)
        }
    
    def check_quality(self, dataset):
        """Sprawdza jakość danych"""
        issues = {
            'too_short': 0,
            'too_long': 0,
            'formatting_errors': 0,
            'encoding_issues': 0
        }
        
        for item in dataset:
            if len(item['response']) < 20:
                issues['too_short'] += 1
            if len(item['response']) > 2000:
                issues['too_long'] += 1
            # Więcej checków...
            
        return issues
    
    def measure_diversity(self, dataset):
        """Mierzy różnorodność datasetu"""
        # Unikalne pierwsze słowa
        first_words = [d['instruction'].split()[0] for d in dataset]
        diversity_score = len(set(first_words)) / len(first_words)
        
        # Podobieństwo semantyczne
        embeddings = self.get_embeddings([d['instruction'] for d in dataset])
        avg_similarity = self.calculate_avg_similarity(embeddings)
        
        return {
            'lexical_diversity': diversity_score,
            'semantic_diversity': 1 - avg_similarity,
            'unique_instructions': len(set(d['instruction'] for d in dataset))
        }
```

## 3. Proces dostosowywania modeli

### 3.1 OpenAI Fine-tuning API

```python
import openai
import time
import json

class OpenAIFineTuner:
    def __init__(self, api_key):
        openai.api_key = api_key
        
    def prepare_training_file(self, dataset, output_path):
        """Przygotowuje plik w formacie OpenAI"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                conversation = {
                    "messages": [
                        {"role": "system", "content": "Jesteś specjalistycznym asystentem."},
                        {"role": "user", "content": item['instruction']},
                        {"role": "assistant", "content": item['response']}
                    ]
                }
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    def upload_file(self, file_path):
        """Uploaduje plik treningowy"""
        with open(file_path, 'rb') as f:
            response = openai.File.create(
                file=f,
                purpose='fine-tune'
            )
        return response['id']
    
    def start_fine_tuning(self, training_file_id, model="gpt-3.5-turbo"):
        """Rozpoczyna proces fine-tuningu"""
        response = openai.FineTuningJob.create(
            training_file=training_file_id,
            model=model,
            hyperparameters={
                "n_epochs": 3,
                "learning_rate_multiplier": 0.1,
                "batch_size": 4
            }
        )
        return response['id']
    
    def monitor_training(self, job_id):
        """Monitoruje postęp treningu"""
        while True:
            response = openai.FineTuningJob.retrieve(job_id)
            status = response['status']
            
            print(f"Status: {status}")
            
            if status == 'succeeded':
                return response['fine_tuned_model']
            elif status == 'failed':
                raise Exception(f"Training failed: {response.get('error')}")
                
            time.sleep(60)  # Sprawdzaj co minutę
    
    def test_fine_tuned_model(self, model_name, test_prompts):
        """Testuje fine-tunowany model"""
        results = []
        
        for prompt in test_prompts:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            results.append({
                'prompt': prompt,
                'response': response.choices[0].message.content
            })
            
        return results

# Przykład użycia
fine_tuner = OpenAIFineTuner(api_key="your-api-key")

# 1. Przygotuj dane
dataset = load_your_dataset()
fine_tuner.prepare_training_file(dataset, "training_data.jsonl")

# 2. Upload
file_id = fine_tuner.upload_file("training_data.jsonl")

# 3. Rozpocznij trening
job_id = fine_tuner.start_fine_tuning(file_id)

# 4. Monitoruj
model_name = fine_tuner.monitor_training(job_id)

# 5. Testuj
test_results = fine_tuner.test_fine_tuned_model(model_name, test_prompts)
```

### 3.2 Hugging Face Transformers

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset

class HuggingFaceFineTuner:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Dodaj padding token jeśli nie istnieje
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, data):
        """Przygotowuje dataset do treningu"""
        def formatting_func(examples):
            texts = []
            for instruction, response in zip(examples['instruction'], examples['response']):
                text = f"### Instrukcja:\n{instruction}\n\n### Odpowiedź:\n{response}"
                texts.append(text)
            return {'text': texts}
        
        # Konwertuj do Dataset
        dataset = Dataset.from_dict({
            'instruction': [d['instruction'] for d in data],
            'response': [d['response'] for d in data]
        })
        
        # Formatuj
        dataset = dataset.map(formatting_func, batched=True)
        
        # Tokenizuj
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def setup_training(self, output_dir="./fine-tuned-model"):
        """Konfiguruje parametry treningu"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            learning_rate=5e-5,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True,  # Używaj mixed precision
            gradient_checkpointing=True,  # Oszczędność pamięci
            report_to="tensorboard"
        )
        
        return training_args
    
    def train(self, train_dataset, eval_dataset=None):
        """Przeprowadza trening"""
        training_args = self.setup_training()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, nie masked
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Trening
        trainer.train()
        
        # Zapisz model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
    def inference(self, prompt, max_length=200):
        """Generuje odpowiedź używając fine-tunowanego modelu"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Przykład użycia z PEFT (Parameter Efficient Fine-Tuning)
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_model(model):
    """Konfiguruje LoRA dla efektywnego fine-tuningu"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Warstwy do fine-tuningu
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model
```

### 3.3 Parametry i hiperparametry

```python
class HyperparameterOptimizer:
    def __init__(self):
        self.search_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
            'batch_size': [2, 4, 8, 16],
            'num_epochs': [1, 3, 5],
            'warmup_ratio': [0.0, 0.1, 0.2],
            'weight_decay': [0.0, 0.01, 0.1]
        }
        
    def grid_search(self, train_func, eval_func, dataset):
        """Przeprowadza grid search"""
        results = []
        
        for lr in self.search_space['learning_rate']:
            for bs in self.search_space['batch_size']:
                for epochs in self.search_space['num_epochs']:
                    config = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'num_epochs': epochs
                    }
                    
                    # Trenuj model
                    model = train_func(dataset, config)
                    
                    # Ewaluuj
                    score = eval_func(model)
                    
                    results.append({
                        'config': config,
                        'score': score
                    })
                    
        # Znajdź najlepszą konfigurację
        best = max(results, key=lambda x: x['score'])
        return best
    
    def analyze_training_dynamics(self, training_history):
        """Analizuje dynamikę treningu"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(training_history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        
        # Learning rate
        axes[0, 1].plot(training_history['learning_rate'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('LR')
        
        # Gradient norm
        if 'grad_norm' in training_history:
            axes[1, 0].plot(training_history['grad_norm'])
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Norm')
        
        # Validation metrics
        if 'eval_loss' in training_history:
            axes[1, 1].plot(training_history['eval_loss'])
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].set_xlabel('Eval Steps')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        return fig
```

## 4. Techniki zaawansowane

### 4.1 Instruction Tuning

```python
class InstructionTuner:
    def __init__(self):
        self.instruction_templates = [
            "Wykonaj następujące zadanie: {task}",
            "Instrukcja: {task}\nOdpowiedź:",
            "Potrzebuję pomocy z: {task}",
            "Proszę {task}",
            "{task}"
        ]
    
    def create_instruction_dataset(self, tasks_and_responses):
        """Tworzy dataset w formacie instrukcyjnym"""
        instruction_data = []
        
        for task, response in tasks_and_responses:
            # Użyj różnych templates
            template = random.choice(self.instruction_templates)
            instruction = template.format(task=task)
            
            instruction_data.append({
                'instruction': instruction,
                'response': response,
                'task_type': self.classify_task(task)
            })
            
        return instruction_data
    
    def classify_task(self, task):
        """Klasyfikuje typ zadania"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['napisz', 'stwórz', 'wygeneruj']):
            return 'generation'
        elif any(word in task_lower for word in ['odpowiedz', 'wyjaśnij', 'czym']):
            return 'qa'
        elif any(word in task_lower for word in ['przetłumacz', 'tłumaczenie']):
            return 'translation'
        elif any(word in task_lower for word in ['podsumuj', 'streść']):
            return 'summarization'
        else:
            return 'other'
```

### 4.2 Multi-task Learning

```python
class MultiTaskFineTuner:
    def __init__(self):
        self.task_prefixes = {
            'classification': "[KLASYFIKUJ]",
            'generation': "[GENERUJ]",
            'translation': "[TŁUMACZ]",
            'summarization': "[PODSUMUJ]",
            'qa': "[ODPOWIEDZ]"
        }
    
    def prepare_multitask_data(self, datasets_by_task):
        """Przygotowuje dane dla multi-task learningu"""
        combined_data = []
        
        for task_type, dataset in datasets_by_task.items():
            prefix = self.task_prefixes[task_type]
            
            for item in dataset:
                combined_data.append({
                    'instruction': f"{prefix} {item['instruction']}",
                    'response': item['response'],
                    'task_type': task_type
                })
        
        # Shuffle żeby mieszać zadania
        random.shuffle(combined_data)
        return combined_data
    
    def create_task_weighted_sampler(self, dataset, weights):
        """Tworzy sampler z wagami dla różnych zadań"""
        from torch.utils.data import WeightedRandomSampler
        
        # Oblicz wagi dla każdego przykładu
        sample_weights = []
        for item in dataset:
            task_type = item['task_type']
            weight = weights.get(task_type, 1.0)
            sample_weights.append(weight)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        return sampler
```

### 4.3 Continuous Learning

```python
class ContinuousLearner:
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.version_history = []
        
    def incremental_update(self, new_data, version_name):
        """Aktualizuje model nowymi danymi bez catastrophic forgetting"""
        # Załaduj aktualny model
        model = self.load_latest_model()
        
        # Przygotuj dane
        # Mixing: 80% nowe dane, 20% replay starych
        replay_data = self.sample_replay_data(size=len(new_data) * 0.25)
        combined_data = new_data + replay_data
        
        # Fine-tune z mniejszym learning rate
        training_args = TrainingArguments(
            learning_rate=1e-5,  # Mniejszy LR dla incremental
            num_train_epochs=1,  # Mniej epok
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_strategy="epoch"
        )
        
        # Trenuj
        trainer = self.setup_trainer(model, combined_data, training_args)
        trainer.train()
        
        # Zapisz nową wersję
        self.save_version(model, version_name)
        
    def sample_replay_data(self, size):
        """Próbkuje dane z poprzednich wersji"""
        if not self.version_history:
            return []
            
        # Załaduj dane z poprzednich wersji
        all_historical_data = []
        for version in self.version_history[-3:]:  # Ostatnie 3 wersje
            data = self.load_version_data(version)
            all_historical_data.extend(data)
        
        # Próbkuj
        if len(all_historical_data) > size:
            return random.sample(all_historical_data, size)
        return all_historical_data
```

## 5. Przykład end-to-end: Chatbot prawny

```python
class LegalChatbotFineTuning:
    def __init__(self):
        self.data_processor = DataPreprocessor()
        self.augmenter = DataAugmenter()
        self.validator = DatasetValidator()
        
    def full_pipeline(self, raw_data_path, output_model_path):
        """Kompletny pipeline fine-tuningu chatbota prawnego"""
        
        print("1. Ładowanie surowych danych...")
        raw_data = self.load_legal_qa_data(raw_data_path)
        
        print("2. Czyszczenie i preprocessing...")
        cleaned_data, stats = self.data_processor.clean_dataset(raw_data)
        print(f"   Usunięto {stats['removed']} przykładów")
        
        print("3. Augmentacja danych...")
        augmented_data = self.augment_legal_data(cleaned_data)
        print(f"   Dataset powiększony do {len(augmented_data)} przykładów")
        
        print("4. Walidacja jakości...")
        validation_report = self.validator.comprehensive_validation(augmented_data)
        self.print_validation_report(validation_report)
        
        print("5. Podział na train/val/test...")
        train_data, val_data, test_data = self.split_data(augmented_data)
        
        print("6. Przygotowanie do treningu...")
        training_file = self.prepare_openai_format(train_data, "legal_train.jsonl")
        validation_file = self.prepare_openai_format(val_data, "legal_val.jsonl")
        
        print("7. Upload plików...")
        train_file_id = self.upload_to_openai(training_file)
        val_file_id = self.upload_to_openai(validation_file)
        
        print("8. Rozpoczęcie fine-tuningu...")
        job_id = self.start_legal_finetuning(train_file_id, val_file_id)
        
        print("9. Monitorowanie treningu...")
        model_name = self.monitor_and_wait(job_id)
        
        print("10. Ewaluacja na zbiorze testowym...")
        test_results = self.evaluate_legal_model(model_name, test_data)
        
        print("11. Generowanie raportu końcowego...")
        self.generate_final_report(model_name, test_results)
        
        return model_name
    
    def augment_legal_data(self, data):
        """Specyficzna augmentacja dla domeny prawnej"""
        augmented = []
        
        legal_variations = {
            "Co mówi prawo o": ["Jakie są przepisy dotyczące", "Jak reguluje prawo"],
            "Czy mogę": ["Czy jest dozwolone", "Czy prawo pozwala"],
            "Jakie są konsekwencje": ["Jakie są kary za", "Co grozi za"]
        }
        
        for item in data:
            # Oryginał
            augmented.append(item)
            
            # Wariacje
            instruction = item['instruction']
            for pattern, replacements in legal_variations.items():
                if pattern in instruction:
                    for replacement in replacements:
                        new_instruction = instruction.replace(pattern, replacement)
                        augmented.append({
                            'instruction': new_instruction,
                            'response': item['response']
                        })
                        
        return augmented
    
    def evaluate_legal_model(self, model_name, test_data):
        """Ewaluacja specyficzna dla domeny prawnej"""
        results = {
            'accuracy_metrics': {},
            'legal_compliance': {},
            'safety_checks': {},
            'example_outputs': []
        }
        
        for test_item in test_data[:50]:  # Testuj na 50 przykładach
            response = self.get_model_response(model_name, test_item['instruction'])
            
            # Sprawdź dokładność prawną
            legal_accuracy = self.check_legal_accuracy(response, test_item['response'])
            
            # Sprawdź bezpieczeństwo (brak porad prawnych)
            safety_score = self.check_legal_safety(response)
            
            # Sprawdź cytowania
            citation_quality = self.check_citations(response)
            
            results['example_outputs'].append({
                'instruction': test_item['instruction'],
                'expected': test_item['response'],
                'generated': response,
                'scores': {
                    'legal_accuracy': legal_accuracy,
                    'safety': safety_score,
                    'citations': citation_quality
                }
            })
            
        return results
```

## 6. Troubleshooting i optymalizacja

### 6.1 Częste problemy i rozwiązania

```python
class FineTuningTroubleshooter:
    def diagnose_training_issues(self, training_logs):
        """Diagnozuje problemy z treningiem"""
        issues = []
        
        # Overfitting
        if self.detect_overfitting(training_logs):
            issues.append({
                'problem': 'Overfitting detected',
                'symptoms': 'Training loss decreasing, validation loss increasing',
                'solutions': [
                    'Zwiększ dropout',
                    'Dodaj więcej danych',
                    'Zmniejsz liczbę epok',
                    'Użyj regularizacji'
                ]
            })
        
        # Underfitting
        if self.detect_underfitting(training_logs):
            issues.append({
                'problem': 'Underfitting detected',
                'symptoms': 'Both training and validation loss high',
                'solutions': [
                    'Zwiększ liczbę epok',
                    'Zwiększ learning rate',
                    'Użyj większego modelu',
                    'Sprawdź jakość danych'
                ]
            })
        
        # Catastrophic forgetting
        if self.detect_catastrophic_forgetting(training_logs):
            issues.append({
                'problem': 'Catastrophic forgetting',
                'symptoms': 'Model forgot original capabilities',
                'solutions': [
                    'Zmniejsz learning rate',
                    'Dodaj dane z original task',
                    'Użyj elastic weight consolidation',
                    'Zastosuj LoRA/QLoRA'
                ]
            })
        
        return issues
    
    def optimize_memory_usage(self):
        """Optymalizuje użycie pamięci podczas treningu"""
        optimizations = {
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'gradient_accumulation_steps': 4,
            'per_device_batch_size': 1,
            'optim': 'adamw_8bit',  # 8-bit Adam
            'load_in_8bit': True,
            'use_lora': True
        }
        
        return optimizations
```

### 6.2 Monitorowanie jakości

```python
class QualityMonitor:
    def __init__(self):
        self.metrics = {
            'perplexity': [],
            'response_diversity': [],
            'factual_accuracy': [],
            'style_consistency': []
        }
    
    def continuous_evaluation(self, model, test_set, interval=100):
        """Ciągła ewaluacja podczas treningu"""
        def evaluate_callback(trainer, step):
            if step % interval == 0:
                # Oblicz metryki
                metrics = self.calculate_metrics(trainer.model, test_set)
                
                # Zapisz
                for metric_name, value in metrics.items():
                    self.metrics[metric_name].append({
                        'step': step,
                        'value': value
                    })
                
                # Alert jeśli degradacja
                if self.detect_quality_degradation():
                    self.send_alert("Quality degradation detected!")
                    
        return evaluate_callback
    
    def detect_quality_degradation(self, threshold=0.1):
        """Wykrywa degradację jakości"""
        if len(self.metrics['perplexity']) < 2:
            return False
            
        latest = self.metrics['perplexity'][-1]['value']
        previous = self.metrics['perplexity'][-2]['value']
        
        return (latest - previous) / previous > threshold
```

## 7. Ćwiczenia praktyczne

### Ćwiczenie 1: Przygotowanie danych
1. Weź 100 przykładów Q&A z twojej domeny
2. Oczyść dane (usuń duplikaty, popraw formatowanie)
3. Augmentuj do 500 przykładów
4. Waliduj jakość datasetu
5. Podziel na train/val/test (70/15/15)

### Ćwiczenie 2: Fine-tuning małego modelu
1. Wybierz mały model (np. GPT-2, Phi-2)
2. Przeprowadź fine-tuning na swoich danych
3. Monitoruj loss i metryki
4. Porównaj z modelem bazowym
5. Zoptymalizuj hiperparametry

### Ćwiczenie 3: A/B Testing
1. Stwórz 2 wersje fine-tuningu:
   - Wersja A: tylko twoje dane
   - Wersja B: twoje dane + augmentacja
2. Ewaluuj oba modele
3. Przeprowadź blind test z użytkownikami
4. Przeanalizuj wyniki

### Ćwiczenie 4: Debugging
1. Celowo wprowadź problemy:
   - Zbyt mały dataset (20 przykładów)
   - Zbyt duży learning rate
   - Złe formatowanie danych
2. Zidentyfikuj problemy z logów
3. Zastosuj odpowiednie poprawki
4. Dokumentuj proces

## 8. Podsumowanie i najlepsze praktyki

### 8.1 Checklist przed fine-tuningiem

- [ ] Czy prompt engineering nie wystarczy?
- [ ] Czy mam min. 500-1000 wysokiej jakości przykładów?
- [ ] Czy dane są zróżnicowane i reprezentatywne?
- [ ] Czy mam dane walidacyjne i testowe?
- [ ] Czy określiłem metryki sukcesu?
- [ ] Czy mam budżet na eksperymenty?
- [ ] Czy mam plan utrzymania modelu?

### 8.2 Do's and Don'ts

**DO:**
- ✅ Zacznij od małego datasetu i iteruj
- ✅ Monitoruj metryki podczas treningu
- ✅ Zachowaj dane testowe na końcową ewaluację
- ✅ Używaj technik regularizacji
- ✅ Dokumentuj wszystkie eksperymenty

**DON'T:**
- ❌ Nie trenuj na danych testowych
- ❌ Nie ignoruj validation loss
- ❌ Nie używaj jednej metryki do oceny
- ❌ Nie zapomnij o data privacy
- ❌ Nie deployuj bez thorough testing

## 9. Zasoby i dalsza nauka

### Narzędzia:
- **OpenAI Fine-tuning UI** - Web interface dla fine-tuningu
- **Weights & Biases** - Tracking eksperymentów
- **Hugging Face AutoTrain** - No-code fine-tuning
- **LangSmith** - Monitoring i ewaluacja LLM

### Literatura:
- "Fine-Tuning Language Models from Human Preferences" (2019)
- "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)
- "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

### Kursy:
- "Fine-tuning Large Language Models" - DeepLearning.AI
- "Efficient Fine-Tuning of LLMs" - Hugging Face Course
- "Advanced NLP with Transformers" - O'Reilly