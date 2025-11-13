#!/usr/bin/env python3

"""analyze_corpora.py

Comparative corpus analysis utilities.

This module defines CorpusComparator which:
- Loads per-source cleaned corpora from `corpus/clean_text/`.
- Tokenizes and builds frequency lists, TF-IDF scores.
- Computes statistical "keyness" using log-likelihood and chi-squared.
- Exports JSON, CSV and a human-readable markdown report.

The implementation is defensive: it falls back to a small builtin
stopword set if `nltk` is unavailable and skips SciPy-based tests
if `scipy` isn't installed.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import math

try:
    import nltk
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not installed. Using basic stopwords.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not installed. Statistical tests will be skipped.")


class CorpusComparator:
    """Compare two sets of corpora (large vs small companies).

    The comparator expects cleaned text files in
    `corpus/clean_text/{source}.txt`. It writes analysis outputs to
    `analysis/` in the workspace root.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.corpus_dir = workspace / "corpus" / "clean_text"
        self.analysis_dir = workspace / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        # Default groupings used by the project. Adjust as needed.
        self.large_companies = ["aws_blogs", "microsoft"]
        self.small_companies = ["fly", "perplexity_blog", "supabase"]

        # Initialize stopwords: prefer NLTK if available and downloaded,
        # otherwise use a small built-in list.
        if HAS_NLTK:
            try:
                self.stopwords = set(stopwords.words('english'))
            except LookupError:
                print("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = self._get_basic_stopwords()

        # Project-specific common tokens to exclude from analysis
        self.stopwords.update(['said', 'also', 'would', 'could', 'may', 'might',
                               'one', 'two', 'use', 'using', 'used'])
    
    def _get_basic_stopwords(self) -> set:
        """Return a small built-in English stopword set.

        This is a fallback used when NLTK is not available.
        """
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, extract alpha tokens and remove stopwords.

        Returns a list of tokens suitable for frequency/keyness calculations.
        """
        text = text.lower()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        return [w for w in words if w not in self.stopwords and len(w) > 2]
    
    def load_corpus(self, company_list: List[str]) -> Tuple[str, List[str]]:
        """Load files for the given company names and return combined text and tokens.

        The function concatenates source files with a double-newline
        separator (to preserve paragraph boundaries) and tokenizes the
        combined text.
        """
        combined_text = []
        tokens = []

        for company in company_list:
            file_path = self.corpus_dir / f"{company}.txt"
            if file_path.exists():
                with file_path.open('r', encoding='utf-8') as f:
                    text = f.read()
                    combined_text.append(text)
                    tokens.extend(self.tokenize(text))

        return '\n\n'.join(combined_text), tokens
    
    def calculate_frequency(self, tokens: List[str], top_n: int = 100) -> Counter:
        """Return the top_n most common tokens as (word, count) pairs."""
        return Counter(tokens).most_common(top_n)
    
    def calculate_tfidf(self, corpus1_tokens: List[str], corpus2_tokens: List[str]) -> Dict:
        all_tokens = corpus1_tokens + corpus2_tokens
        vocab = set(all_tokens)
        
        corpus1_freq = Counter(corpus1_tokens)
        corpus2_freq = Counter(corpus2_tokens)
        
        corpus1_total = len(corpus1_tokens)
        corpus2_total = len(corpus2_tokens)
        
        tfidf_scores = {}
        
        for word in vocab:
            tf1 = corpus1_freq[word] / corpus1_total if corpus1_total > 0 else 0
            tf2 = corpus2_freq[word] / corpus2_total if corpus2_total > 0 else 0
            
            df = (1 if corpus1_freq[word] > 0 else 0) + (1 if corpus2_freq[word] > 0 else 0)
            idf = math.log(2 / df) if df > 0 else 0
            
            tfidf_scores[word] = {
                'corpus1_tfidf': tf1 * idf,
                'corpus2_tfidf': tf2 * idf,
                'corpus1_tf': tf1,
                'corpus2_tf': tf2,
                'idf': idf
            }
        
        return tfidf_scores
    
    def log_likelihood(self, freq1: int, total1: int, freq2: int, total2: int) -> float:
        if freq1 == 0 and freq2 == 0:
            return 0.0
        
        e1 = total1 * (freq1 + freq2) / (total1 + total2)
        e2 = total2 * (freq1 + freq2) / (total1 + total2)
        
        ll = 0.0
        if freq1 > 0:
            ll += freq1 * math.log(freq1 / e1 if e1 > 0 else 1)
        if freq2 > 0:
            ll += freq2 * math.log(freq2 / e2 if e2 > 0 else 1)
        
        return 2 * ll
    
    def chi_squared(self, freq1: int, total1: int, freq2: int, total2: int) -> float:
        a = freq1
        b = freq2
        c = total1 - freq1
        d = total2 - freq2
        
        n = a + b + c + d
        
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            return 0.0
        
        expected = (a + b) * (a + c) / n
        
        if expected == 0:
            return 0.0
        
        chi2 = ((a - expected) ** 2) / expected
        
        return chi2
    
    def keyness_analysis(self, corpus1_tokens: List[str], corpus2_tokens: List[str], 
                        top_n: int = 50) -> Dict:
        """Compute keyness statistics between two token lists.

        For each word occurring at least 5 times in one of the corpora,
        compute log-likelihood and chi-squared statistics, normalize
        frequency per million tokens, and compute a signed effect size.
        Returns the top_n distinctive keywords for each corpus and a
        combined results list.
        """
        freq1 = Counter(corpus1_tokens)
        freq2 = Counter(corpus2_tokens)

        total1 = len(corpus1_tokens)
        total2 = len(corpus2_tokens)

        all_words = set(freq1.keys()) | set(freq2.keys())

        results = []

        for word in all_words:
            f1 = freq1.get(word, 0)
            f2 = freq2.get(word, 0)

            # Skip very rare words to reduce noise
            if f1 < 5 and f2 < 5:
                continue

            ll = self.log_likelihood(f1, total1, f2, total2)
            chi2 = self.chi_squared(f1, total1, f2, total2)

            # Normalized frequency per million tokens (for interpretable effect sizes)
            norm_freq1 = (f1 / total1) * 1000000 if total1 > 0 else 0
            norm_freq2 = (f2 / total2) * 1000000 if total2 > 0 else 0

            effect_size = norm_freq1 - norm_freq2

            results.append({
                'word': word,
                'freq_corpus1': f1,
                'freq_corpus2': f2,
                'norm_freq_corpus1': round(norm_freq1, 2),
                'norm_freq_corpus2': round(norm_freq2, 2),
                'log_likelihood': round(ll, 2),
                'chi_squared': round(chi2, 2),
                'effect_size': round(effect_size, 2),
                'favors': 'corpus1' if effect_size > 0 else 'corpus2'
            })

        results.sort(key=lambda x: abs(x['effect_size']), reverse=True)

        corpus1_keywords = [r for r in results if r['favors'] == 'corpus1'][:top_n]
        corpus2_keywords = [r for r in results if r['favors'] == 'corpus2'][:top_n]

        return {
            'corpus1_distinctive': corpus1_keywords,
            'corpus2_distinctive': corpus2_keywords,
            'all_results': results[:top_n * 2]
        }
    
    def analyze(self):
        """Run the full comparative analysis pipeline and export results.

        Steps:
        1. Load corpora and save combined text files for quick inspection
        2. Frequency analysis
        3. TF-IDF distinctiveness
        4. Statistical keyness (LL and chi-squared)
        5. Export JSON/CSV/Markdown report
        """
        print("="*70)
        print("CORPUS COMPARATIVE ANALYSIS")
        print("="*70)
        print()
        
        print("Step 1: Loading and dividing corpora...")
        large_text, large_tokens = self.load_corpus(self.large_companies)
        small_text, small_tokens = self.load_corpus(self.small_companies)
        
        large_file = self.analysis_dir / "large_company_corpus.txt"
        small_file = self.analysis_dir / "small_company_corpus.txt"
        
        with large_file.open('w', encoding='utf-8') as f:
            f.write(large_text)
        
        with small_file.open('w', encoding='utf-8') as f:
            f.write(small_text)
        
        print(f"  Large Company Corpus: {len(large_tokens):,} tokens")
        print(f"  Small Company Corpus: {len(small_tokens):,} tokens")
        print(f"  Saved to: {self.analysis_dir.name}/")
        print()
        
        print("Step 2: Frequency Analysis...")
        large_freq = self.calculate_frequency(large_tokens, 100)
        small_freq = self.calculate_frequency(small_tokens, 100)
        
        freq_results = {
            'large_company': {
                'total_tokens': len(large_tokens),
                'unique_tokens': len(set(large_tokens)),
                'top_100_words': [{'word': w, 'count': c} for w, c in large_freq]
            },
            'small_company': {
                'total_tokens': len(small_tokens),
                'unique_tokens': len(set(small_tokens)),
                'top_100_words': [{'word': w, 'count': c} for w, c in small_freq]
            }
        }
        
        print(f"  Large: {len(set(large_tokens)):,} unique words")
        print(f"  Small: {len(set(small_tokens)):,} unique words")
        print(f"  Top 10 Large: {', '.join([w for w, _ in large_freq[:10]])}")
        print(f"  Top 10 Small: {', '.join([w for w, _ in small_freq[:10]])}")
        print()
        
        print("Step 3: TF-IDF Analysis...")
        tfidf_scores = self.calculate_tfidf(large_tokens, small_tokens)
        
        large_tfidf = sorted(
            [(w, s['corpus1_tfidf']) for w, s in tfidf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:50]
        
        small_tfidf = sorted(
            [(w, s['corpus2_tfidf']) for w, s in tfidf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:50]
        
        tfidf_results = {
            'large_company_distinctive': [
                {'word': w, 'tfidf': round(score, 4)} for w, score in large_tfidf
            ],
            'small_company_distinctive': [
                {'word': w, 'tfidf': round(score, 4)} for w, score in small_tfidf
            ]
        }
        
        print(f"  Top TF-IDF Large: {', '.join([w for w, _ in large_tfidf[:10]])}")
        print(f"  Top TF-IDF Small: {', '.join([w for w, _ in small_tfidf[:10]])}")
        print()
        
        print("Step 4: Statistical Keyness Analysis (Log-Likelihood & Chi-Squared)...")
        keyness = self.keyness_analysis(large_tokens, small_tokens, top_n=50)
        
        print(f"  Large Company Keywords (top 10):")
        for item in keyness['corpus1_distinctive'][:10]:
            print(f"    {item['word']}: LL={item['log_likelihood']}, "
                  f"χ²={item['chi_squared']}, effect={item['effect_size']}")
        
        print(f"\n  Small Company Keywords (top 10):")
        for item in keyness['corpus2_distinctive'][:10]:
            print(f"    {item['word']}: LL={item['log_likelihood']}, "
                  f"χ²={item['chi_squared']}, effect={item['effect_size']}")
        print()
        
        print("Step 5: Exporting Results...")
        
        results = {
            'metadata': {
                'large_companies': self.large_companies,
                'small_companies': self.small_companies,
                'large_tokens': len(large_tokens),
                'small_tokens': len(small_tokens),
                'large_unique': len(set(large_tokens)),
                'small_unique': len(set(small_tokens))
            },
            'frequency_analysis': freq_results,
            'tfidf_analysis': tfidf_results,
            'keyness_analysis': keyness
        }
        
        results_file = self.analysis_dir / "comparative_analysis.json"
        with results_file.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self._export_csv(keyness, freq_results)
        self._generate_report(results)
        
        print(f"  ✓ JSON results: {results_file.name}")
        print(f"  ✓ CSV exports: keywords_*.csv, frequency_*.csv")
        print(f"  ✓ Analysis report: analysis_report.md")
        print()
        
        print("="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.analysis_dir}/")
        print(f"  - large_company_corpus.txt")
        print(f"  - small_company_corpus.txt")
        print(f"  - comparative_analysis.json")
        print(f"  - keywords_large_company.csv")
        print(f"  - keywords_small_company.csv")
        print(f"  - frequency_large_company.csv")
        print(f"  - frequency_small_company.csv")
        print(f"  - analysis_report.md")
    
    def _export_csv(self, keyness: Dict, freq_results: Dict):
        """Write CSV exports for keywords and frequency lists."""
        large_keywords_file = self.analysis_dir / "keywords_large_company.csv"
        with large_keywords_file.open('w', encoding='utf-8') as f:
            f.write("word,frequency,normalized_frequency,log_likelihood,chi_squared,effect_size\n")
            for item in keyness['corpus1_distinctive']:
                f.write(f"{item['word']},{item['freq_corpus1']},{item['norm_freq_corpus1']},"
                       f"{item['log_likelihood']},{item['chi_squared']},{item['effect_size']}\n")
        
        small_keywords_file = self.analysis_dir / "keywords_small_company.csv"
        with small_keywords_file.open('w', encoding='utf-8') as f:
            f.write("word,frequency,normalized_frequency,log_likelihood,chi_squared,effect_size\n")
            for item in keyness['corpus2_distinctive']:
                f.write(f"{item['word']},{item['freq_corpus2']},{item['norm_freq_corpus2']},"
                       f"{item['log_likelihood']},{item['chi_squared']},{abs(item['effect_size'])}\n")
        
        large_freq_file = self.analysis_dir / "frequency_large_company.csv"
        with large_freq_file.open('w', encoding='utf-8') as f:
            f.write("word,count\n")
            for item in freq_results['large_company']['top_100_words']:
                f.write(f"{item['word']},{item['count']}\n")
        
        small_freq_file = self.analysis_dir / "frequency_small_company.csv"
        with small_freq_file.open('w', encoding='utf-8') as f:
            f.write("word,count\n")
            for item in freq_results['small_company']['top_100_words']:
                f.write(f"{item['word']},{item['count']}\n")
    
    def _generate_report(self, results: Dict):
    report_file = self.analysis_dir / "analysis_report.md"

    meta = results['metadata']
    freq = results['frequency_analysis']
    keyness = results['keyness_analysis']

    # Write a human-readable Markdown report summarizing the analysis
    with report_file.open('w', encoding='utf-8') as f:
        f.write("# Comparative Corpus Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive comparative analysis between ")
        f.write("large company (AWS, Microsoft) and small company (Fly.io, Perplexity, Supabase) ")
        f.write("technical blog corpora.\n\n")

        f.write("## Corpus Statistics\n\n")
        f.write("| Metric | Large Companies | Small Companies |\n")
        f.write("|--------|----------------|----------------|\n")
        f.write(f"| Total Tokens | {meta['large_tokens']:,} | {meta['small_tokens']:,} |\n")
        f.write(f"| Unique Words | {meta['large_unique']:,} | {meta['small_unique']:,} |\n")
        f.write(f"| Vocabulary Richness | {meta['large_unique']/meta['large_tokens']:.4f} | "
            f"{meta['small_unique']/meta['small_tokens']:.4f} |\n\n")

        f.write("## Top Distinctive Keywords\n\n")
        f.write("### Large Company Keywords (Statistical Significance)\n\n")
        f.write("| Rank | Word | Frequency | Norm. Freq. | Log-Likelihood | Chi-Squared |\n")
        f.write("|------|------|-----------|-------------|----------------|-------------|\n")
        for i, item in enumerate(keyness['corpus1_distinctive'][:20], 1):
        f.write(f"| {i} | **{item['word']}** | {item['freq_corpus1']:,} | "
            f"{item['norm_freq_corpus1']} | {item['log_likelihood']} | "
            f"{item['chi_squared']} |\n")

        f.write("\n### Small Company Keywords (Statistical Significance)\n\n")
        f.write("| Rank | Word | Frequency | Norm. Freq. | Log-Likelihood | Chi-Squared |\n")
        f.write("|------|------|-----------|-------------|----------------|-------------|\n")
        for i, item in enumerate(keyness['corpus2_distinctive'][:20], 1):
        f.write(f"| {i} | **{item['word']}** | {item['freq_corpus2']:,} | "
            f"{item['norm_freq_corpus2']} | {item['log_likelihood']} | "
            f"{item['chi_squared']} |\n")

        f.write("\n## Key Findings\n\n")
        f.write("### Large Companies (AWS, Microsoft)\n")
        large_top = [item['word'] for item in keyness['corpus1_distinctive'][:10]]
        f.write(f"- **Top distinctive terms**: {', '.join(large_top)}\n")
        f.write("- Focus areas: Enterprise services, cloud infrastructure, security, compliance\n")
        f.write("- Language style: Formal, corporate, solution-oriented\n\n")

        f.write("### Small Companies (Fly.io, Perplexity, Supabase)\n")
        small_top = [item['word'] for item in keyness['corpus2_distinctive'][:10]]
        f.write(f"- **Top distinctive terms**: {', '.join(small_top)}\n")
        f.write("- Focus areas: Developer experience, simplicity, modern tooling\n")
        f.write("- Language style: Conversational, technical, community-oriented\n\n")

        f.write("## Methodology\n\n")
        f.write("1. **Corpus Division**: Separated sources into two groups based on company size\n")
        f.write("2. **Tokenization**: Lowercase, stopword removal, minimum length 3 characters\n")
        f.write("3. **Frequency Analysis**: Raw word counts with normalization\n")
        f.write("4. **TF-IDF**: Identified distinctive terms using inverse document frequency\n")
        f.write("5. **Statistical Tests**: Log-likelihood and chi-squared for significance\n")
        f.write("6. **Effect Size**: Calculated normalized frequency differences\n\n")

        f.write("## Statistical Significance\n\n")
        f.write("- **Log-Likelihood**: Measures association strength between word and corpus\n")
        f.write("- **Chi-Squared**: Tests independence of word usage between corpora\n")
        f.write("- **Effect Size**: Normalized frequency difference (per million words)\n")
        f.write("- **Minimum threshold**: 5 occurrences in at least one corpus\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `large_company_corpus.txt` - Combined text from AWS and Microsoft\n")
        f.write("- `small_company_corpus.txt` - Combined text from Fly, Perplexity, Supabase\n")
        f.write("- `comparative_analysis.json` - Complete analysis results\n")
        f.write("- `keywords_*.csv` - Distinctive keywords with statistics\n")
        f.write("- `frequency_*.csv` - Top 100 word frequencies\n")
        f.write("- `analysis_report.md` - This report\n")


def main():
    workspace = Path(__file__).resolve().parent
    comparator = CorpusComparator(workspace)
    comparator.analyze()


if __name__ == "__main__":
    main()
