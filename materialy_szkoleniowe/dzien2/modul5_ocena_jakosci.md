# Moduł 5: Metody oceny jakości modeli językowych

## Cel modułu
Po zakończeniu tego modułu uczestnik będzie:
- Znał i stosował metryki automatycznej oceny jakości
- Projektował i przeprowadzał ewaluację z udziałem ludzi
- Oceniał skuteczność fine-tuningu i promptów
- Implementował kompleksowe systemy ewaluacji

## 1. Metryki automatyczne

### 1.1 Perpleksja (Perplexity)

**Definicja**: Miara tego, jak "zaskoczony" jest model przez tekst. Niższa perpleksja = lepszy model.

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityCalculator:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
    def calculate_perplexity(self, text):
        """Oblicza perpleksję dla danego tekstu"""
        # Tokenizacja
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        # Perpleksja = exp(loss)
        perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def calculate_dataset_perplexity(self, texts, batch_size=8):
        """Oblicza średnią perpleksję dla zbioru tekstów"""
        perplexities = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_perplexities = [self.calculate_perplexity(text) for text in batch]
            perplexities.extend(batch_perplexities)
            
        return {
            'mean_perplexity': np.mean(perplexities),
            'std_perplexity': np.std(perplexities),
            'min_perplexity': np.min(perplexities),
            'max_perplexity': np.max(perplexities)
        }
    
    def compare_models(self, model_names, test_texts):
        """Porównuje perpleksję różnych modeli"""
        results = {}
        
        for model_name in model_names:
            print(f"Evaluating {model_name}...")
            calculator = PerplexityCalculator(model_name)
            results[model_name] = calculator.calculate_dataset_perplexity(test_texts)
            
        return results

# Przykład użycia
calculator = PerplexityCalculator("gpt2")
text = "Sztuczna inteligencja rewolucjonizuje sposób w jaki pracujemy."
perplexity = calculator.calculate_perplexity(text)
print(f"Perpleksja: {perplexity:.2f}")
```

### 1.2 BLEU (Bilingual Evaluation Understudy)

**Zastosowanie**: Ocena jakości tłumaczeń i generowania tekstu przez porównanie z referencją.

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk

class BLEUEvaluator:
    def __init__(self):
        # Pobierz potrzebne zasoby NLTK
        nltk.download('punkt', quiet=True)
        self.smoothing = SmoothingFunction()
        
    def calculate_bleu(self, reference, candidate, n_gram=4):
        """Oblicza BLEU score dla pojedynczej pary zdań"""
        # Tokenizacja
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # BLEU wymaga referencji jako listy list
        reference_list = [reference_tokens]
        
        # Oblicz BLEU z różnymi n-gramami
        scores = {}
        for n in range(1, n_gram + 1):
            weight = tuple(1.0 if i < n else 0.0 for i in range(n_gram))
            score = sentence_bleu(
                reference_list, 
                candidate_tokens,
                weights=weight,
                smoothing_function=self.smoothing.method1
            )
            scores[f'bleu_{n}'] = score
            
        # Oblicz też ogólny BLEU
        scores['bleu_combined'] = sentence_bleu(
            reference_list,
            candidate_tokens,
            smoothing_function=self.smoothing.method1
        )
        
        return scores
    
    def evaluate_generation_quality(self, references, candidates):
        """Ewaluuje jakość generowania na zbiorze danych"""
        if len(references) != len(candidates):
            raise ValueError("Liczba referencji i kandydatów musi być równa")
            
        all_scores = []
        
        for ref, cand in zip(references, candidates):
            scores = self.calculate_bleu(ref, cand)
            all_scores.append(scores)
            
        # Agreguj wyniki
        aggregated = {}
        for metric in all_scores[0].keys():
            values = [score[metric] for score in all_scores]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return aggregated
    
    def detailed_analysis(self, reference, candidate):
        """Szczegółowa analiza BLEU z wizualizacją"""
        from collections import Counter
        
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        # N-gram overlap analysis
        analysis = {
            'length_ratio': len(cand_tokens) / len(ref_tokens),
            'exact_matches': sum(1 for t in cand_tokens if t in ref_tokens),
            'n_gram_overlaps': {}
        }
        
        for n in range(1, 5):
            ref_ngrams = Counter(zip(*[ref_tokens[i:] for i in range(n)]))
            cand_ngrams = Counter(zip(*[cand_tokens[i:] for i in range(n)]))
            
            overlap = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            
            analysis['n_gram_overlaps'][f'{n}-gram'] = {
                'overlap': overlap,
                'total': total,
                'precision': overlap / total if total > 0 else 0
            }
            
        return analysis

# Przykład użycia
evaluator = BLEUEvaluator()

reference = "Kot siedzi na macie i obserwuje ptaki za oknem"
candidate = "Kot leży na dywanie i patrzy na ptaki przez okno"

scores = evaluator.calculate_bleu(reference, candidate)
print("BLEU Scores:", scores)

analysis = evaluator.detailed_analysis(reference, candidate)
print("\nSzczegółowa analiza:", analysis)
```

### 1.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Zastosowanie**: Ocena jakości streszczeń przez porównanie z referencyjnymi streszczeniami.

```python
from rouge_score import rouge_scorer
import pandas as pd

class ROUGEEvaluator:
    def __init__(self, rouge_types=['rouge1', 'rouge2', 'rougeL']):
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        self.rouge_types = rouge_types
        
    def calculate_rouge(self, reference, candidate):
        """Oblicza ROUGE scores dla pary tekstów"""
        scores = self.scorer.score(reference, candidate)
        
        # Formatuj wyniki
        formatted_scores = {}
        for rouge_type in self.rouge_types:
            formatted_scores[rouge_type] = {
                'precision': scores[rouge_type].precision,
                'recall': scores[rouge_type].recall,
                'f1': scores[rouge_type].fmeasure
            }
            
        return formatted_scores
    
    def evaluate_summarization(self, references, summaries):
        """Ewaluuje jakość streszczeń"""
        all_scores = []
        
        for ref, summ in zip(references, summaries):
            scores = self.calculate_rouge(ref, summ)
            all_scores.append(scores)
            
        # Agregacja
        df_scores = pd.DataFrame(all_scores)
        aggregated = {}
        
        for rouge_type in self.rouge_types:
            rouge_data = pd.DataFrame([s[rouge_type] for s in all_scores])
            aggregated[rouge_type] = {
                'precision': {
                    'mean': rouge_data['precision'].mean(),
                    'std': rouge_data['precision'].std()
                },
                'recall': {
                    'mean': rouge_data['recall'].mean(),
                    'std': rouge_data['recall'].std()
                },
                'f1': {
                    'mean': rouge_data['f1'].mean(),
                    'std': rouge_data['f1'].std()
                }
            }
            
        return aggregated
    
    def compare_summarization_models(self, reference_summaries, model_outputs):
        """Porównuje różne modele streszczające"""
        comparison_results = {}
        
        for model_name, summaries in model_outputs.items():
            print(f"Evaluating {model_name}...")
            results = self.evaluate_summarization(reference_summaries, summaries)
            comparison_results[model_name] = results
            
        # Twórz ranking
        ranking = []
        for model_name, results in comparison_results.items():
            avg_f1 = np.mean([
                results[rouge]['f1']['mean'] 
                for rouge in self.rouge_types
            ])
            ranking.append((model_name, avg_f1))
            
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return comparison_results, ranking

# Przykład użycia
evaluator = ROUGEEvaluator()

reference_summary = """
Sztuczna inteligencja transformuje biznes poprzez automatyzację procesów,
poprawę decyzji i tworzenie nowych możliwości. Kluczowe jest odpowiedzialne
wdrażanie z uwzględnieniem etyki i bezpieczeństwa.
"""

generated_summary = """
AI rewolucjonizuje przedsiębiorstwa przez automatyzację i wspomaganie decyzji.
Ważne jest etyczne podejście do implementacji.
"""

scores = evaluator.calculate_rouge(reference_summary, generated_summary)
print("ROUGE Scores:")
for rouge_type, values in scores.items():
    print(f"{rouge_type}: P={values['precision']:.3f}, R={values['recall']:.3f}, F1={values['f1']:.3f}")
```

### 1.4 BERTScore

**Zastosowanie**: Semantyczna ocena podobieństwa używając embeddings z BERT.

```python
from bert_score import BERTScorer
import torch

class BERTScoreEvaluator:
    def __init__(self, model_type='bert-base-multilingual-cased', lang='pl'):
        self.scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=True
        )
        
    def calculate_bertscore(self, references, candidates):
        """Oblicza BERTScore dla par tekstów"""
        P, R, F1 = self.scorer.score(candidates, references)
        
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist(),
            'aggregated': {
                'precision_mean': P.mean().item(),
                'recall_mean': R.mean().item(),
                'f1_mean': F1.mean().item(),
                'precision_std': P.std().item(),
                'recall_std': R.std().item(),
                'f1_std': F1.std().item()
            }
        }
    
    def visualize_alignment(self, reference, candidate):
        """Wizualizuje alignment między tekstami"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Oblicz similarity matrix
        P, R, F1 = self.scorer.score([candidate], [reference], return_hash=False)
        
        # Tokenizacja
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        # Twórz heatmapę
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Placeholder dla similarity matrix (w praktyce użyj prawdziwej)
        similarity_matrix = torch.rand(len(cand_tokens), len(ref_tokens))
        
        sns.heatmap(
            similarity_matrix.numpy(),
            xticklabels=ref_tokens,
            yticklabels=cand_tokens,
            cmap='Blues',
            ax=ax
        )
        
        ax.set_xlabel('Reference tokens')
        ax.set_ylabel('Candidate tokens')
        ax.set_title('Token Similarity Heatmap')
        
        return fig
    
    def threshold_analysis(self, references, candidates, thresholds=[0.7, 0.8, 0.9]):
        """Analiza progowa dla BERTScore"""
        scores = self.calculate_bertscore(references, candidates)
        f1_scores = scores['f1']
        
        analysis = {}
        for threshold in thresholds:
            above_threshold = sum(1 for score in f1_scores if score >= threshold)
            percentage = (above_threshold / len(f1_scores)) * 100
            
            analysis[f'threshold_{threshold}'] = {
                'count': above_threshold,
                'percentage': percentage
            }
            
        return analysis

# Przykład użycia
evaluator = BERTScoreEvaluator()

references = [
    "Model skutecznie generuje odpowiedzi na pytania.",
    "System analizuje dane i wyciąga wnioski."
]

candidates = [
    "Model efektywnie odpowiada na zapytania.",
    "System przetwarza informacje i formułuje konkluzje."
]

scores = evaluator.calculate_bertscore(references, candidates)
print("BERTScore Results:")
print(f"Mean F1: {scores['aggregated']['f1_mean']:.3f} (±{scores['aggregated']['f1_std']:.3f})")
```

## 2. Human Evaluation (Ocena przez ludzi)

### 2.1 Metodologie oceny

```python
class HumanEvaluationFramework:
    def __init__(self):
        self.evaluation_criteria = {
            'relevance': {
                'question': 'Czy odpowiedź jest związana z pytaniem?',
                'scale': [1, 5],
                'anchors': {
                    1: 'Zupełnie niezwiązana',
                    3: 'Częściowo związana',
                    5: 'Całkowicie związana'
                }
            },
            'coherence': {
                'question': 'Czy odpowiedź jest spójna i logiczna?',
                'scale': [1, 5],
                'anchors': {
                    1: 'Chaotyczna, nielogiczna',
                    3: 'Częściowo spójna',
                    5: 'Całkowicie spójna'
                }
            },
            'helpfulness': {
                'question': 'Czy odpowiedź jest pomocna?',
                'scale': [1, 5],
                'anchors': {
                    1: 'Zupełnie niepomocna',
                    3: 'Średnio pomocna',
                    5: 'Bardzo pomocna'
                }
            },
            'accuracy': {
                'question': 'Czy informacje są poprawne?',
                'scale': [1, 5],
                'anchors': {
                    1: 'Zawiera błędy',
                    3: 'Częściowo poprawna',
                    5: 'Całkowicie poprawna'
                }
            }
        }
        
    def create_evaluation_interface(self, examples):
        """Tworzy interfejs do oceny przez ludzi"""
        import ipywidgets as widgets
        from IPython.display import display
        
        evaluations = []
        
        for idx, example in enumerate(examples):
            print(f"\n--- Przykład {idx + 1} ---")
            print(f"Pytanie: {example['question']}")
            print(f"Odpowiedź: {example['response']}")
            
            example_scores = {'example_id': idx}
            
            for criterion, config in self.evaluation_criteria.items():
                slider = widgets.IntSlider(
                    value=3,
                    min=config['scale'][0],
                    max=config['scale'][1],
                    step=1,
                    description=criterion,
                    style={'description_width': 'initial'}
                )
                
                display(widgets.HTML(f"<b>{config['question']}</b>"))
                display(slider)
                
                example_scores[criterion] = slider
                
            evaluations.append(example_scores)
            
        return evaluations
    
    def calculate_inter_rater_agreement(self, ratings):
        """Oblicza zgodność między oceniającymi (Inter-rater agreement)"""
        from sklearn.metrics import cohen_kappa_score
        import itertools
        
        # Konwertuj ratings do formatu: {rater: [scores]}
        raters = list(ratings.keys())
        
        if len(raters) < 2:
            return {"error": "Potrzeba minimum 2 oceniających"}
            
        # Oblicz Cohen's Kappa dla każdej pary
        kappa_scores = {}
        
        for rater1, rater2 in itertools.combinations(raters, 2):
            ratings1 = ratings[rater1]
            ratings2 = ratings[rater2]
            
            # Upewnij się że oceniali te same przykłady
            common_examples = set(ratings1.keys()) & set(ratings2.keys())
            
            scores1 = [ratings1[ex] for ex in common_examples]
            scores2 = [ratings2[ex] for ex in common_examples]
            
            kappa = cohen_kappa_score(scores1, scores2, weights='quadratic')
            kappa_scores[f"{rater1}_vs_{rater2}"] = kappa
            
        # Oblicz średnie Kappa
        avg_kappa = np.mean(list(kappa_scores.values()))
        
        return {
            'pairwise_kappa': kappa_scores,
            'average_kappa': avg_kappa,
            'interpretation': self.interpret_kappa(avg_kappa)
        }
    
    def interpret_kappa(self, kappa):
        """Interpretacja wartości Kappa"""
        if kappa < 0:
            return "Brak zgodności"
        elif kappa < 0.20:
            return "Słaba zgodność"
        elif kappa < 0.40:
            return "Umiarkowana zgodność"
        elif kappa < 0.60:
            return "Średnia zgodność"
        elif kappa < 0.80:
            return "Znaczna zgodność"
        else:
            return "Prawie idealna zgodność"
    
    def design_a_b_test(self, model_a_outputs, model_b_outputs, sample_size=100):
        """Projektuje test A/B dla porównania modeli"""
        import random
        
        # Przygotuj pary do porównania
        comparison_pairs = []
        
        for idx, (output_a, output_b) in enumerate(zip(model_a_outputs, model_b_outputs)):
            # Randomizuj kolejność
            if random.random() > 0.5:
                pair = {
                    'id': idx,
                    'option_1': output_a,
                    'option_2': output_b,
                    'correct_order': 'A_B'
                }
            else:
                pair = {
                    'id': idx,
                    'option_1': output_b,
                    'option_2': output_a,
                    'correct_order': 'B_A'
                }
                
            comparison_pairs.append(pair)
            
        # Wybierz próbkę
        if len(comparison_pairs) > sample_size:
            comparison_pairs = random.sample(comparison_pairs, sample_size)
            
        return comparison_pairs
    
    def analyze_a_b_results(self, test_results):
        """Analizuje wyniki testu A/B"""
        model_a_wins = 0
        model_b_wins = 0
        ties = 0
        
        for result in test_results:
            if result['preference'] == 'option_1':
                if result['correct_order'] == 'A_B':
                    model_a_wins += 1
                else:
                    model_b_wins += 1
            elif result['preference'] == 'option_2':
                if result['correct_order'] == 'A_B':
                    model_b_wins += 1
                else:
                    model_a_wins += 1
            else:
                ties += 1
                
        total = len(test_results)
        
        # Test statystyczny
        from scipy.stats import binomtest
        
        # Test czy Model A jest lepszy od B
        binom_result = binomtest(
            model_a_wins,
            model_a_wins + model_b_wins,
            p=0.5,
            alternative='greater'
        )
        
        return {
            'model_a_preference': (model_a_wins / total) * 100,
            'model_b_preference': (model_b_wins / total) * 100,
            'ties': (ties / total) * 100,
            'statistical_significance': binom_result.pvalue < 0.05,
            'p_value': binom_result.pvalue,
            'confidence_interval': [
                binom_result.proportion_ci(confidence_level=0.95).low,
                binom_result.proportion_ci(confidence_level=0.95).high
            ]
        }
```

### 2.2 Skale oceny i ankiety

```python
class EvaluationSurveyBuilder:
    def __init__(self):
        self.survey_templates = {}
        
    def create_likert_scale(self, question, scale_points=5):
        """Tworzy pytanie ze skalą Likerta"""
        if scale_points == 5:
            options = [
                "Zdecydowanie się nie zgadzam",
                "Nie zgadzam się",
                "Neutralny",
                "Zgadzam się",
                "Zdecydowanie się zgadzam"
            ]
        elif scale_points == 7:
            options = [
                "Zdecydowanie się nie zgadzam",
                "Nie zgadzam się",
                "Raczej się nie zgadzam",
                "Neutralny",
                "Raczej się zgadzam",
                "Zgadzam się",
                "Zdecydowanie się zgadzam"
            ]
        else:
            options = [str(i) for i in range(1, scale_points + 1)]
            
        return {
            'type': 'likert',
            'question': question,
            'options': options,
            'scale_points': scale_points
        }
    
    def create_comparison_task(self, context, option_a, option_b):
        """Tworzy zadanie porównawcze"""
        return {
            'type': 'comparison',
            'context': context,
            'options': {
                'A': option_a,
                'B': option_b
            },
            'questions': [
                {
                    'id': 'preference',
                    'text': 'Która odpowiedź jest lepsza?',
                    'type': 'choice',
                    'options': ['A', 'B', 'Obie równie dobre', 'Obie słabe']
                },
                {
                    'id': 'confidence',
                    'text': 'Jak pewny jesteś swojej oceny?',
                    'type': 'scale',
                    'min': 1,
                    'max': 5,
                    'labels': {1: 'Niepewny', 5: 'Bardzo pewny'}
                }
            ]
        }
    
    def create_quality_assessment(self, text_to_evaluate):
        """Tworzy kompleksową ocenę jakości"""
        return {
            'type': 'quality_assessment',
            'text': text_to_evaluate,
            'dimensions': [
                {
                    'id': 'fluency',
                    'name': 'Płynność',
                    'description': 'Czy tekst jest gramatycznie poprawny i naturalny?',
                    'scale': [1, 5]
                },
                {
                    'id': 'relevance',
                    'name': 'Trafność',
                    'description': 'Czy odpowiedź odnosi się do pytania?',
                    'scale': [1, 5]
                },
                {
                    'id': 'informativeness',
                    'name': 'Informatywność',
                    'description': 'Czy odpowiedź dostarcza użytecznych informacji?',
                    'scale': [1, 5]
                },
                {
                    'id': 'factuality',
                    'name': 'Faktyczność',
                    'description': 'Czy informacje są prawdziwe i dokładne?',
                    'scale': [1, 5],
                    'requires_expertise': True
                }
            ]
        }
    
    def export_to_form(self, survey, platform='google_forms'):
        """Eksportuje ankietę do formatu platformy"""
        if platform == 'google_forms':
            return self._export_to_google_forms(survey)
        elif platform == 'typeform':
            return self._export_to_typeform(survey)
        elif platform == 'csv':
            return self._export_to_csv(survey)
        else:
            raise ValueError(f"Nieznana platforma: {platform}")
```

## 3. Ocena fine-tuningu i promptów

### 3.1 Porównanie przed i po fine-tuningu

```python
class FineTuningEvaluator:
    def __init__(self, base_model, finetuned_model, test_dataset):
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.test_dataset = test_dataset
        
    def comprehensive_comparison(self):
        """Przeprowadza kompleksowe porównanie modeli"""
        results = {
            'quantitative_metrics': self.compare_quantitative_metrics(),
            'qualitative_analysis': self.compare_qualitative_aspects(),
            'performance_metrics': self.compare_performance(),
            'error_analysis': self.analyze_errors(),
            'improvement_areas': self.identify_improvements()
        }
        
        return results
    
    def compare_quantitative_metrics(self):
        """Porównuje metryki ilościowe"""
        metrics = {}
        
        # Perpleksja
        base_perplexity = self.calculate_perplexity(self.base_model)
        ft_perplexity = self.calculate_perplexity(self.finetuned_model)
        
        metrics['perplexity'] = {
            'base': base_perplexity,
            'finetuned': ft_perplexity,
            'improvement': ((base_perplexity - ft_perplexity) / base_perplexity) * 100
        }
        
        # Dokładność na zadaniach
        base_accuracy = self.evaluate_task_accuracy(self.base_model)
        ft_accuracy = self.evaluate_task_accuracy(self.finetuned_model)
        
        metrics['accuracy'] = {
            'base': base_accuracy,
            'finetuned': ft_accuracy,
            'improvement': ft_accuracy - base_accuracy
        }
        
        # BLEU/ROUGE dla generacji
        base_generation = self.evaluate_generation_quality(self.base_model)
        ft_generation = self.evaluate_generation_quality(self.finetuned_model)
        
        metrics['generation_quality'] = {
            'base': base_generation,
            'finetuned': ft_generation
        }
        
        return metrics
    
    def compare_qualitative_aspects(self):
        """Porównuje aspekty jakościowe"""
        sample_size = min(50, len(self.test_dataset))
        samples = random.sample(self.test_dataset, sample_size)
        
        comparisons = []
        
        for sample in samples:
            base_response = self.generate_response(self.base_model, sample['input'])
            ft_response = self.generate_response(self.finetuned_model, sample['input'])
            
            comparison = {
                'input': sample['input'],
                'expected': sample.get('expected_output', ''),
                'base_response': base_response,
                'finetuned_response': ft_response,
                'analysis': self.analyze_responses(base_response, ft_response, sample)
            }
            
            comparisons.append(comparison)
            
        return comparisons
    
    def analyze_responses(self, base_response, ft_response, sample):
        """Analizuje różnice między odpowiedziami"""
        analysis = {
            'length_difference': len(ft_response) - len(base_response),
            'style_consistency': self.check_style_consistency(ft_response, sample),
            'domain_specific_terms': self.count_domain_terms(ft_response) - self.count_domain_terms(base_response),
            'follows_format': self.check_format_compliance(ft_response, sample)
        }
        
        return analysis
    
    def identify_improvements(self):
        """Identyfikuje obszary poprawy"""
        improvements = {
            'response_format': [],
            'domain_knowledge': [],
            'consistency': [],
            'edge_cases': []
        }
        
        # Analiza formatowania odpowiedzi
        format_improvements = self.analyze_format_improvements()
        improvements['response_format'] = format_improvements
        
        # Analiza wiedzy domenowej
        domain_improvements = self.analyze_domain_knowledge()
        improvements['domain_knowledge'] = domain_improvements
        
        return improvements
    
    def create_evaluation_report(self):
        """Tworzy raport z ewaluacji"""
        comparison = self.comprehensive_comparison()
        
        report = f"""
# Raport Ewaluacji Fine-Tuningu

## Podsumowanie
- Model bazowy: {self.base_model.name}
- Model fine-tunowany: {self.finetuned_model.name}
- Liczba przykładów testowych: {len(self.test_dataset)}

## Metryki Ilościowe

### Perpleksja
- Model bazowy: {comparison['quantitative_metrics']['perplexity']['base']:.2f}
- Model fine-tunowany: {comparison['quantitative_metrics']['perplexity']['finetuned']:.2f}
- Poprawa: {comparison['quantitative_metrics']['perplexity']['improvement']:.1f}%

### Dokładność
- Model bazowy: {comparison['quantitative_metrics']['accuracy']['base']:.1f}%
- Model fine-tunowany: {comparison['quantitative_metrics']['accuracy']['finetuned']:.1f}%
- Poprawa: +{comparison['quantitative_metrics']['accuracy']['improvement']:.1f} punktów procentowych

## Analiza Jakościowa

### Główne obszary poprawy:
"""
        for area, improvements in comparison['improvement_areas'].items():
            if improvements:
                report += f"\n**{area}**:\n"
                for improvement in improvements[:3]:
                    report += f"- {improvement}\n"
                    
        return report
```

### 3.2 Ewaluacja promptów

```python
class PromptEvaluator:
    def __init__(self, model):
        self.model = model
        self.evaluation_criteria = {
            'effectiveness': 'Czy prompt generuje pożądane odpowiedzi?',
            'consistency': 'Czy odpowiedzi są spójne przy wielokrotnych wywołaniach?',
            'efficiency': 'Czy prompt jest optymalny pod względem długości?',
            'robustness': 'Czy prompt radzi sobie z edge cases?'
        }
        
    def evaluate_prompt_variations(self, prompt_variants, test_cases):
        """Porównuje różne warianty promptów"""
        results = {}
        
        for variant_name, prompt_template in prompt_variants.items():
            variant_results = {
                'scores': {},
                'examples': [],
                'statistics': {}
            }
            
            # Testuj na wszystkich przypadkach
            for test_case in test_cases:
                prompt = prompt_template.format(**test_case['variables'])
                response = self.model.generate(prompt)
                
                # Oceń odpowiedź
                scores = self.score_response(response, test_case['expected'])
                
                variant_results['examples'].append({
                    'input': test_case,
                    'prompt': prompt,
                    'response': response,
                    'scores': scores
                })
                
            # Oblicz statystyki
            variant_results['statistics'] = self.calculate_statistics(
                variant_results['examples']
            )
            
            results[variant_name] = variant_results
            
        return results
    
    def score_response(self, response, expected):
        """Ocenia odpowiedź według kryteriów"""
        scores = {}
        
        # Skuteczność - czy odpowiedź spełnia oczekiwania
        scores['effectiveness'] = self.calculate_effectiveness_score(response, expected)
        
        # Długość - czy odpowiedź ma odpowiednią długość
        scores['length_appropriateness'] = self.calculate_length_score(response, expected)
        
        # Formatowanie - czy odpowiedź ma właściwy format
        scores['format_compliance'] = self.check_format_compliance(response, expected)
        
        return scores
    
    def test_prompt_robustness(self, prompt_template, edge_cases):
        """Testuje odporność promptu na edge cases"""
        robustness_results = {
            'handled_well': [],
            'partial_failures': [],
            'complete_failures': []
        }
        
        for edge_case in edge_cases:
            try:
                prompt = prompt_template.format(**edge_case['variables'])
                response = self.model.generate(prompt)
                
                # Sprawdź jakość odpowiedzi
                quality = self.assess_edge_case_handling(response, edge_case)
                
                result = {
                    'case': edge_case,
                    'response': response,
                    'quality': quality
                }
                
                if quality >= 0.8:
                    robustness_results['handled_well'].append(result)
                elif quality >= 0.5:
                    robustness_results['partial_failures'].append(result)
                else:
                    robustness_results['complete_failures'].append(result)
                    
            except Exception as e:
                robustness_results['complete_failures'].append({
                    'case': edge_case,
                    'error': str(e)
                })
                
        return robustness_results
    
    def optimize_prompt(self, initial_prompt, test_cases, iterations=5):
        """Iteracyjnie optymalizuje prompt"""
        current_prompt = initial_prompt
        optimization_history = []
        
        for i in range(iterations):
            # Ewaluuj obecny prompt
            results = self.evaluate_single_prompt(current_prompt, test_cases)
            
            optimization_history.append({
                'iteration': i,
                'prompt': current_prompt,
                'score': results['overall_score'],
                'details': results
            })
            
            # Jeśli wynik zadowalający, zakończ
            if results['overall_score'] > 0.9:
                break
                
            # Generuj sugestie poprawy
            suggestions = self.generate_improvement_suggestions(results)
            
            # Zastosuj sugestie
            current_prompt = self.apply_suggestions(current_prompt, suggestions)
            
        return {
            'final_prompt': current_prompt,
            'history': optimization_history,
            'improvement': optimization_history[-1]['score'] - optimization_history[0]['score']
        }
```

## 4. Automatyczne systemy ewaluacji

### 4.1 Framework do ciągłej ewaluacji

```python
class ContinuousEvaluationSystem:
    def __init__(self, models_to_track):
        self.models = models_to_track
        self.metrics_history = []
        self.alert_thresholds = {
            'perplexity_increase': 0.1,
            'accuracy_decrease': 0.05,
            'response_time_increase': 0.2
        }
        
    def run_evaluation_cycle(self):
        """Przeprowadza cykl ewaluacji"""
        timestamp = datetime.now()
        cycle_results = {
            'timestamp': timestamp,
            'models': {}
        }
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            model_results = {
                'performance_metrics': self.evaluate_performance(model),
                'quality_metrics': self.evaluate_quality(model),
                'safety_metrics': self.evaluate_safety(model),
                'user_satisfaction': self.get_user_satisfaction(model_name)
            }
            
            cycle_results['models'][model_name] = model_results
            
        self.metrics_history.append(cycle_results)
        
        # Sprawdź alerty
        alerts = self.check_for_alerts(cycle_results)
        if alerts:
            self.send_alerts(alerts)
            
        return cycle_results
    
    def evaluate_performance(self, model):
        """Ewaluuje wydajność modelu"""
        test_prompts = self.get_test_prompts()
        
        latencies = []
        tokens_per_second = []
        
        for prompt in test_prompts:
            start_time = time.time()
            response = model.generate(prompt)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            # Oblicz tokens/s
            response_tokens = len(self.tokenize(response))
            tps = response_tokens / latency
            tokens_per_second.append(tps)
            
        return {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'avg_tokens_per_second': np.mean(tokens_per_second),
            'throughput': len(test_prompts) / sum(latencies)
        }
    
    def create_evaluation_dashboard(self):
        """Tworzy dashboard z wynikami"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Przygotuj dane
        timestamps = [r['timestamp'] for r in self.metrics_history]
        
        # Twórz subploty
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Perpleksja', 'Dokładność', 'Czas odpowiedzi', 'Satysfakcja użytkowników')
        )
        
        # Dla każdego modelu
        for model_name in self.models.keys():
            # Perpleksja
            perplexities = [
                r['models'][model_name]['quality_metrics']['perplexity'] 
                for r in self.metrics_history
            ]
            fig.add_trace(
                go.Scatter(x=timestamps, y=perplexities, name=f'{model_name} - Perplexity'),
                row=1, col=1
            )
            
            # Dokładność
            accuracies = [
                r['models'][model_name]['quality_metrics']['accuracy'] 
                for r in self.metrics_history
            ]
            fig.add_trace(
                go.Scatter(x=timestamps, y=accuracies, name=f'{model_name} - Accuracy'),
                row=1, col=2
            )
            
            # Czas odpowiedzi
            latencies = [
                r['models'][model_name]['performance_metrics']['avg_latency'] 
                for r in self.metrics_history
            ]
            fig.add_trace(
                go.Scatter(x=timestamps, y=latencies, name=f'{model_name} - Latency'),
                row=2, col=1
            )
            
            # Satysfakcja
            satisfaction = [
                r['models'][model_name]['user_satisfaction']['score'] 
                for r in self.metrics_history
            ]
            fig.add_trace(
                go.Scatter(x=timestamps, y=satisfaction, name=f'{model_name} - Satisfaction'),
                row=2, col=2
            )
            
        fig.update_layout(height=800, showlegend=True)
        return fig
    
    def generate_evaluation_report(self, period='weekly'):
        """Generuje raport z ewaluacji"""
        latest_results = self.metrics_history[-1]
        
        report = {
            'executive_summary': self.create_executive_summary(period),
            'detailed_metrics': self.compile_detailed_metrics(period),
            'trends': self.analyze_trends(period),
            'recommendations': self.generate_recommendations(),
            'visualizations': self.create_visualizations(period)
        }
        
        return report
```

### 4.2 A/B Testing Framework

```python
class ABTestingFramework:
    def __init__(self):
        self.active_tests = {}
        self.completed_tests = []
        
    def create_test(self, test_name, variant_a, variant_b, metrics, duration_hours=168):
        """Tworzy nowy test A/B"""
        test = {
            'name': test_name,
            'variants': {
                'A': variant_a,
                'B': variant_b
            },
            'metrics': metrics,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=duration_hours),
            'allocation': {'A': 0.5, 'B': 0.5},  # 50/50 split
            'results': {
                'A': [],
                'B': []
            }
        }
        
        self.active_tests[test_name] = test
        return test
    
    def route_request(self, test_name, user_id):
        """Przydziela użytkownika do wariantu"""
        if test_name not in self.active_tests:
            return None
            
        test = self.active_tests[test_name]
        
        # Deterministyczne przydzielenie na podstawie user_id
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        if (hash_value % 100) < (test['allocation']['A'] * 100):
            return 'A'
        else:
            return 'B'
    
    def record_metric(self, test_name, variant, metric_name, value):
        """Zapisuje metrykę dla wariantu"""
        if test_name not in self.active_tests:
            return
            
        test = self.active_tests[test_name]
        test['results'][variant].append({
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.now()
        })
    
    def analyze_test(self, test_name):
        """Analizuje wyniki testu"""
        if test_name not in self.active_tests:
            return None
            
        test = self.active_tests[test_name]
        
        analysis = {
            'test_name': test_name,
            'duration': (datetime.now() - test['start_time']).days,
            'sample_sizes': {
                'A': len(test['results']['A']),
                'B': len(test['results']['B'])
            },
            'metrics': {}
        }
        
        # Analiza każdej metryki
        for metric in test['metrics']:
            a_values = [r['value'] for r in test['results']['A'] if r['metric'] == metric]
            b_values = [r['value'] for r in test['results']['B'] if r['metric'] == metric]
            
            if a_values and b_values:
                # Test statystyczny
                from scipy.stats import ttest_ind
                
                t_stat, p_value = ttest_ind(a_values, b_values)
                
                analysis['metrics'][metric] = {
                    'variant_A': {
                        'mean': np.mean(a_values),
                        'std': np.std(a_values),
                        'n': len(a_values)
                    },
                    'variant_B': {
                        'mean': np.mean(b_values),
                        'std': np.std(b_values),
                        'n': len(b_values)
                    },
                    'difference': np.mean(b_values) - np.mean(a_values),
                    'relative_difference': ((np.mean(b_values) - np.mean(a_values)) / np.mean(a_values)) * 100,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'winner': 'B' if np.mean(b_values) > np.mean(a_values) else 'A'
                }
                
        return analysis
    
    def calculate_sample_size(self, baseline_rate, minimum_effect, power=0.8, alpha=0.05):
        """Oblicza wymaganą wielkość próby"""
        from statsmodels.stats.power import tt_ind_solve_power
        
        effect_size = minimum_effect / baseline_rate
        sample_size = tt_ind_solve_power(
            effect_size=effect_size,
            nobs1=None,
            alpha=alpha,
            power=power
        )
        
        return int(np.ceil(sample_size))
```

## 5. Ćwiczenia praktyczne

### Ćwiczenie 1: Implementacja własnej metryki
1. Zdefiniuj metrykę specyficzną dla twojej domeny
2. Zaimplementuj funkcję obliczającą
3. Przetestuj na 10 przykładach
4. Porównaj z istniejącymi metrykami

### Ćwiczenie 2: Projekt Human Evaluation
1. Zaprojektuj ankietę do oceny chatbota
2. Stwórz 20 przykładów do oceny
3. Przeprowadź ewaluację z 5 osobami
4. Oblicz inter-rater agreement
5. Przeanalizuj wyniki

### Ćwiczenie 3: A/B Test promptów
1. Stwórz 2 warianty promptu
2. Przygotuj 50 testowych zapytań
3. Przeprowadź test A/B
4. Przeanalizuj wyniki statystycznie
5. Wybierz zwycięski wariant

### Ćwiczenie 4: Automatyczna ewaluacja
1. Wybierz model do ewaluacji
2. Zaimplementuj 3 metryki automatyczne
3. Stwórz pipeline ewaluacji
4. Uruchom na zbiorze testowym
5. Wygeneruj raport

## 6. Podsumowanie

### Kluczowe wnioski:
1. **Nie ma jednej idealnej metryki** - używaj kombinacji
2. **Human evaluation jest złotym standardem** ale kosztowna
3. **Automatyczne metryki są użyteczne** do szybkiej iteracji
4. **Kontekst ma znaczenie** - dobierz metryki do zastosowania
5. **Ciągła ewaluacja jest kluczowa** dla utrzymania jakości

### Best Practices:
- ✅ Zawsze miej holdout test set
- ✅ Łącz metryki automatyczne z human eval
- ✅ Dokumentuj wszystkie eksperymenty
- ✅ Monitoruj metryki w produkcji
- ✅ Regularnie aktualizuj benchmarki

## Materiały dodatkowe

### Papers:
- "BERTScore: Evaluating Text Generation with BERT" (2019)
- "Human Evaluation of NLG Systems: A Survey" (2020)
- "Evaluating Large Language Models Trained on Code" (2021)
- "Is ChatGPT a Good NLG Evaluator?" (2023)

### Narzędzia:
- **Comet** - Neural metrics for MT
- **SacreBLEU** - Standardized BLEU
- **jury** - Unified evaluation library
- **langcheck** - LLM output evaluation