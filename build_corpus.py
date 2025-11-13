#!/usr/bin/env python3

"""build_corpus.py

Utilities to build a cleaned language corpus from source CSV exports.

This module implements a CorpusBuilder class that:
- Reads CSV files from the workspace root
- Detects which CSV column contains the textual content
- Cleans and normalizes paragraphs (URL/HTML/email removal, unicode fixes)
- Filters boilerplate and low-quality paragraphs by length and token counts
- Deduplicates case-insensitively
- Writes per-source cleaned text and metadata files and a corpus summary

Design notes:
- The pipeline is conservative: it keeps only paragraphs that meet
    minimum length/word thresholds to ensure downstream model quality.
- The implementation is intentionally permissive about missing files or
    unexpected CSV layouts; it attempts to auto-detect the text column.
"""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter
import ftfy
from langdetect import detect, LangDetectException

class CorpusBuilder:
    """Main class that builds a clean corpus from CSV files.

    Usage pattern:
      builder = CorpusBuilder(workspace=Path('.'))
      stats = builder.build()

    The class creates the `corpus/` folder (unless an output_dir is
    provided) and populates `clean_text/`, `metadata/`, and `stats/`.
    """

    def __init__(self, workspace: Path, output_dir: Path = None):
        # Root of the project where CSVs are located
        self.workspace = workspace
        # Output directory for corpus artifacts
        self.output_dir = output_dir or workspace / "corpus"
        self.clean_text_dir = self.output_dir / "clean_text"
        self.metadata_dir = self.output_dir / "metadata"
        self.stats_dir = self.output_dir / "stats"

        # Ensure directories exist (idempotent)
        self.clean_text_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # Precompiled regexes to speed up repeated cleaning
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.html_pattern = re.compile(r'<[^>]+>')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')

        # Simple heuristics for boilerplate — tuned for blog-style exports
        self.boilerplate_patterns = [
            re.compile(r'^(Article Content|By\s+\w+.*?–\s*\w+)', re.IGNORECASE),
            re.compile(r'(Permalink|Comments|Share)\s*$', re.IGNORECASE),
            re.compile(r'^\s*on\s+\d{1,2}\s+[A-Z]{3}\s+\d{4}\s*$', re.IGNORECASE),
            re.compile(r'^\s*in\s+[A-Z].*?(,\s*[A-Z].*?)*\s*$'),
        ]

        # Quality thresholds
        self.min_paragraph_length = 50
        self.min_word_count = 10
        self.max_paragraph_length = 10000
        
    def clean_text(self, text: str) -> str:
        """Normalize a text fragment.

        Steps performed:
        - Fix common unicode issues via ftfy
        - Remove URLs and HTML tags
        - Replace emails with a token
        - Collapse repeated whitespace and remove control chars

        Returns an empty string when input is falsy.
        """
        if not text:
            return ""

        text = ftfy.fix_text(text)
        text = self.url_pattern.sub(' ', text)
        text = self.html_pattern.sub(' ', text)
        text = self.email_pattern.sub('[EMAIL]', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        text = text.strip()

        return text
    
    def is_boilerplate(self, text: str) -> bool:
        """Quick heuristics to decide if a paragraph is boilerplate.

        We treat very short fragments (<20 chars) as boilerplate and also
        scan the paragraph for a set of regex patterns (bylines, nav items).
        """
        if len(text) < 20:
            return True
        for pattern in self.boilerplate_patterns:
            if pattern.search(text):
                return True
        return False
    
    def detect_language(self, text: str) -> str:
        """Detect the language of a text fragment using langdetect.

        For very short text (<30 chars) we return 'unknown' because the
        detector is unreliable on small samples. Any detection errors are
        caught and treated as 'unknown'.
        """
        try:
            return detect(text) if len(text) > 30 else 'unknown'
        except LangDetectException:
            return 'unknown'
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Split a block of text into cleaned paragraphs.

        Processing steps:
        - Normalize newlines and trim
        - Collapse multiple blank lines into a single paragraph separator
        - Clean and filter each candidate paragraph using the cleaning,
          boilerplate and quality-threshold methods

        Returns a list of paragraphs that passed all filters.
        """
        if not text:
            return []

        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.strip().strip('"')

        if not text:
            return []

        text = re.sub(r"\n{3,}", "\n\n", text)
        parts = re.split(r"\n\s*\n", text)

        cleaned = []
        for p in parts:
            p = self.clean_text(p)
            if not p:
                continue

            if self.is_boilerplate(p):
                continue

            if len(p) < self.min_paragraph_length or len(p) > self.max_paragraph_length:
                continue

            word_count = len(p.split())
            if word_count < self.min_word_count:
                continue

            cleaned.append(p)

        return cleaned
    
    def calculate_stats(self, paragraphs: List[str]) -> Dict:
        all_text = ' '.join(paragraphs)
        words = all_text.split()
        
        return {
            'paragraph_count': len(paragraphs),
            'word_count': len(words),
            'char_count': len(all_text),
            'avg_paragraph_length': len(all_text) / len(paragraphs) if paragraphs else 0,
            'avg_words_per_paragraph': len(words) / len(paragraphs) if paragraphs else 0,
            'unique_words': len(set(w.lower() for w in words)),
        }
    
    def detect_text_column(self, header, sample_rows):
        """Attempt to detect which CSV column contains the main text.

        Strategy:
        1. If a header row exists, look for common header names (text, content)
        2. Otherwise, sample up to 200 rows and choose the column with the
           largest average string length (heuristic for text-heavy columns)
        """
        if header:
            for i, h in enumerate(header):
                if h is None:
                    continue
                key = h.strip().lower()
                if key in ("text", "tablescraper-selected-row", "content", "body"):
                    return i

        if not sample_rows:
            return 0

        col_count = max(len(r) for r in sample_rows)
        sums = [0] * col_count
        counts = [0] * col_count

        for r in sample_rows:
            for i in range(len(r)):
                val = r[i]
                if val is not None:
                    sums[i] += len(str(val))
                    counts[i] += 1

        avg = [(sums[i] / counts[i]) if counts[i] else 0 for i in range(col_count)]
        return max(range(col_count), key=lambda i: avg[i])
    
    def process_csv(self, csv_path: Path) -> Tuple[List[str], Dict]:
        """Process a CSV file and extract cleaned paragraphs.

        Returns a tuple: (list_of_paragraphs, metadata_dict).
        The method attempts to auto-detect the text column, extracts
        paragraphs, applies cleaning and deduplication, and computes
        per-source metadata.
        """
        # Read a small sample (first 200 rows) to detect the text column
        paragraphs = []
        sample = []
        header = None

        with csv_path.open("r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = None
            
            for _ in range(200):
                try:
                    row = next(reader)
                    sample.append(row)
                except StopIteration:
                    break
        
        # Choose which column likely holds the article/body text
        text_idx = self.detect_text_column(header, sample)

        with csv_path.open("r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f)
            first_row = None
            try:
                first_row = next(reader)
            except StopIteration:
                pass
            
            has_header = False
            if first_row and any((c is not None and isinstance(c, str) and c.strip().isalpha()) for c in first_row):
                has_header = True
            
            if has_header:
                rows_iter = reader
            else:
                def gen():
                    if first_row is not None:
                        yield first_row
                    for r in reader:
                        yield r
                rows_iter = gen()
            
            for row in rows_iter:
                if not row:
                    continue

                # If the detected text index is beyond current row length
                # fall back to joining the row into a candidate string.
                if text_idx >= len(row):
                    candidate = " ".join(row)
                else:
                    candidate = row[text_idx]

                if candidate is None:
                    continue

                parts = self.extract_paragraphs(str(candidate))
                paragraphs.extend(parts)
        
        # Deduplicate case-insensitively while preserving original text
        seen = set()
        deduped = []
        for p in paragraphs:
            p_lower = p.lower()
            if p_lower in seen:
                continue
            seen.add(p_lower)
            deduped.append(p)
        
        stats = self.calculate_stats(deduped)
        
        sample_text = ' '.join(deduped[:10]) if len(deduped) >= 10 else ' '.join(deduped)
        language = self.detect_language(sample_text)
        
        metadata = {
            'source_file': csv_path.name,
            'source_path': str(csv_path),
            'processed_date': datetime.now().isoformat(),
            'language': language,
            'statistics': stats,
            'cleaning_params': {
                'min_paragraph_length': self.min_paragraph_length,
                'min_word_count': self.min_word_count,
                'max_paragraph_length': self.max_paragraph_length,
                'url_removed': True,
                'html_removed': True,
                'unicode_normalized': True,
                'boilerplate_filtered': True,
                'case_insensitive_dedup': True,
            }
        }
        
        return deduped, metadata
    
    def save_corpus(self, name: str, paragraphs: List[str], metadata: Dict):
        """Write cleaned paragraphs and metadata to disk.

        Text files use double-newline separators between paragraphs.
        Metadata is written as pretty-printed JSON (UTF-8).
        Returns the two Path objects written (text_file, meta_file).
        """
        text_file = self.clean_text_dir / f"{name}.txt"
        meta_file = self.metadata_dir / f"{name}_metadata.json"

        with text_file.open("w", encoding="utf-8") as f:
            for i, p in enumerate(paragraphs):
                f.write(p)
                if i != len(paragraphs) - 1:
                    f.write("\n\n")

        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return text_file, meta_file
    
    def build(self):
        csv_files = sorted(self.workspace.glob("*.csv"))
        all_metadata = {}
        corpus_stats = {
            'total_sources': 0,
            'total_paragraphs': 0,
            'total_words': 0,
            'total_chars': 0,
            'languages': Counter(),
            'sources': []
        }
        
        # Iterate CSV files in workspace root and process each
        print("Building corpus...")
        print(f"Output directory: {self.output_dir}\n")
        
        for csv_path in csv_files:
            if not csv_path.exists():
                continue
            
            print(f"Processing {csv_path.name}...")
            
            paragraphs, metadata = self.process_csv(csv_path)
            name = csv_path.stem
            
            text_file, meta_file = self.save_corpus(name, paragraphs, metadata)
            
            all_metadata[name] = metadata
            corpus_stats['total_sources'] += 1
            corpus_stats['total_paragraphs'] += metadata['statistics']['paragraph_count']
            corpus_stats['total_words'] += metadata['statistics']['word_count']
            corpus_stats['total_chars'] += metadata['statistics']['char_count']
            corpus_stats['languages'][metadata['language']] += 1
            
            corpus_stats['sources'].append({
                'name': name,
                'source_file': csv_path.name,
                'paragraphs': metadata['statistics']['paragraph_count'],
                'words': metadata['statistics']['word_count'],
                'language': metadata['language'],
            })
            
            print(f"  → {metadata['statistics']['paragraph_count']} paragraphs")
            print(f"  → {metadata['statistics']['word_count']:,} words")
            print(f"  → Language: {metadata['language']}")
            print(f"  → Saved to: {text_file.name}\n")
        
        corpus_stats['languages'] = dict(corpus_stats['languages'])
        
        # Persist corpus-level summary stats
        summary_file = self.stats_dir / "corpus_summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(corpus_stats, f, indent=2, ensure_ascii=False)
        
        readme_content = self._generate_readme(corpus_stats)
        readme_file = self.output_dir / "README.md"
        with readme_file.open("w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print("\n" + "="*60)
        print("CORPUS BUILD COMPLETE")
        print("="*60)
        print(f"Total Sources: {corpus_stats['total_sources']}")
        print(f"Total Paragraphs: {corpus_stats['total_paragraphs']:,}")
        print(f"Total Words: {corpus_stats['total_words']:,}")
        print(f"Total Characters: {corpus_stats['total_chars']:,}")
        print(f"Languages: {', '.join(f'{lang}={count}' for lang, count in corpus_stats['languages'].items())}")
        print(f"\nOutput structure:")
        print(f"  {self.clean_text_dir.relative_to(self.workspace)}/  - Clean text files")
        print(f"  {self.metadata_dir.relative_to(self.workspace)}/  - Metadata JSON files")
        print(f"  {self.stats_dir.relative_to(self.workspace)}/     - Corpus statistics")
        print(f"  {readme_file.relative_to(self.workspace)}      - Documentation")
        
        return corpus_stats
    
    def _generate_readme(self, stats: Dict) -> str:
        # Generate a human-readable README summarizing corpus statistics
        readme = f"""# Language Corpus Documentation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This corpus contains cleaned and processed text data from {stats['total_sources']} sources, 
designed for language modeling and NLP research.

## Corpus Statistics

- **Total Sources:** {stats['total_sources']}
- **Total Paragraphs:** {stats['total_paragraphs']:,}
- **Total Words:** {stats['total_words']:,}
- **Total Characters:** {stats['total_chars']:,}
- **Languages:** {', '.join(f'{lang} ({count} sources)' for lang, count in stats['languages'].items())}

## Data Sources

| Source | Paragraphs | Words | Language |
|--------|------------|-------|----------|
"""
        
        for source in stats['sources']:
            readme += f"| {source['name']} | {source['paragraphs']:,} | {source['words']:,} | {source['language']} |\n"
        
        readme += f"""
## Processing Methodology

### Text Cleaning
- **Unicode Normalization:** All text normalized using ftfy to fix encoding issues
- **URL Removal:** All URLs removed and replaced with spaces
- **HTML Tag Removal:** HTML tags stripped from content
- **Email Protection:** Email addresses replaced with `[EMAIL]` token
- **Whitespace Normalization:** Multiple spaces/tabs/newlines collapsed
- **Control Character Removal:** Non-printable characters removed

### Quality Filtering
- **Minimum Paragraph Length:** 50 characters
- **Minimum Word Count:** 10 words per paragraph
- **Maximum Paragraph Length:** 10,000 characters
- **Boilerplate Removal:** Common metadata patterns filtered (dates, bylines, navigation)
- **Deduplication:** Case-insensitive exact duplicate removal

### Language Detection
- Language detection performed using langdetect library
- Detection based on first 10 paragraphs or all available text

## Directory Structure

```
corpus/
├── clean_text/          # Cleaned text files (.txt)
│   ├── source1.txt
│   ├── source2.txt
│   └── ...
├── metadata/            # Per-source metadata (.json)
│   ├── source1_metadata.json
│   ├── source2_metadata.json
│   └── ...
├── stats/              # Corpus-wide statistics
│   └── corpus_summary.json
└── README.md           # This file
```

## File Formats

### Text Files (`clean_text/*.txt`)
- UTF-8 encoded plain text
- One paragraph per block
- Paragraphs separated by double newlines (`\\n\\n`)
- No markup or metadata

### Metadata Files (`metadata/*_metadata.json`)
Each metadata file contains:
- `source_file`: Original CSV filename
- `source_path`: Full path to source file
- `processed_date`: ISO 8601 timestamp
- `language`: Detected language code
- `statistics`: Word counts, paragraph counts, averages
- `cleaning_params`: Parameters used for text cleaning

### Summary Statistics (`stats/corpus_summary.json`)
Aggregated statistics across all sources:
- Total counts (sources, paragraphs, words, characters)
- Language distribution
- Per-source breakdowns

## Usage

### Loading Text Data

```python
from pathlib import Path

corpus_dir = Path("corpus/clean_text")
for text_file in corpus_dir.glob("*.txt"):
    with text_file.open("r", encoding="utf-8") as f:
        content = f.read()
        paragraphs = content.split("\\n\\n")
        # Process paragraphs...
```

### Loading Metadata

```python
import json

metadata_file = Path("corpus/metadata/source_metadata.json")
with metadata_file.open("r", encoding="utf-8") as f:
    metadata = json.load(f)
    print(f"Word count: {{metadata['statistics']['word_count']}}")
```

## Quality Assurance

- All text validated for UTF-8 encoding
- Paragraph length constraints enforced
- Boilerplate and navigation text filtered
- Duplicate paragraphs removed (case-insensitive)
- Language detection for content verification

## Citation

If you use this corpus in your research, please cite:

```
Language Corpus
Generated: {datetime.now().strftime('%Y-%m-%d')}
Sources: {', '.join(s['source_file'] for s in stats['sources'])}
```

## License

Please refer to the original data sources for licensing information.

## Contact

For questions or issues regarding this corpus, please refer to the source 
repository or contact the corpus maintainer.
"""
        
        return readme


def main():
    workspace = Path(__file__).resolve().parent
    builder = CorpusBuilder(workspace)
    builder.build()


if __name__ == "__main__":
    main()
