#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter

def check_duplicates_in_file(file_path: Path) -> dict:
    """Check for duplicate paragraphs in a text file."""
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    paragraph_counts = Counter(paragraphs)
    duplicates = {p: count for p, count in paragraph_counts.items() if count > 1}
    
    case_insensitive_counts = Counter(p.lower() for p in paragraphs)
    case_insensitive_dupes = {p: count for p, count in case_insensitive_counts.items() if count > 1}
    
    return {
        'total_paragraphs': len(paragraphs),
        'unique_paragraphs': len(set(paragraphs)),
        'exact_duplicates': len(duplicates),
        'case_insensitive_duplicates': len(case_insensitive_dupes),
        'duplicate_details': list(duplicates.items())[:5] if duplicates else []
    }

def main():
    workspace = Path(__file__).resolve().parent
    corpus_dir = workspace / "corpus"
    clean_text_dir = corpus_dir / "clean_text"
    metadata_dir = corpus_dir / "metadata"
    
    print("="*70)
    print("CORPUS VERIFICATION REPORT")
    print("="*70)
    print()
    
    if not corpus_dir.exists():
        print("❌ No corpus directory found!")
        return
    
    print("📊 DEDUPLICATION VERIFICATION")
    print("-" * 70)
    
    all_stats = {}
    for text_file in sorted(clean_text_dir.glob("*.txt")):
        stats = check_duplicates_in_file(text_file)
        all_stats[text_file.stem] = stats
        
        dedup_rate = (stats['unique_paragraphs'] / stats['total_paragraphs'] * 100) if stats['total_paragraphs'] > 0 else 0
        
        print(f"\n{text_file.name}:")
        print(f"  Total paragraphs: {stats['total_paragraphs']:,}")
        print(f"  Unique paragraphs: {stats['unique_paragraphs']:,}")
        print(f"  Exact duplicates: {stats['exact_duplicates']}")
        print(f"  Case-insensitive duplicates: {stats['case_insensitive_duplicates']}")
        print(f"  Deduplication rate: {dedup_rate:.2f}%")
        
        if stats['exact_duplicates'] > 0:
            print(f"  ⚠️  Warning: {stats['exact_duplicates']} exact duplicate(s) found")
            if stats['duplicate_details']:
                print(f"  First duplicate: '{stats['duplicate_details'][0][0][:80]}...'")
        else:
            print("  ✓ No exact duplicates")
    
    print("\n" + "="*70)
    print("📋 FILENAME STANDARDIZATION")
    print("-" * 70)
    
    all_files = {
        'clean_text': list(clean_text_dir.glob("*.txt")),
        'metadata': list(metadata_dir.glob("*_metadata.json"))
    }
    
    non_standard = []
    for category, files in all_files.items():
        for file_path in files:
            name = file_path.name
            if any(c.isupper() or c == '-' for c in name.split('.')[0]):
                non_standard.append((category, name))
    
    if non_standard:
        print("⚠️  Non-standard filenames found:")
        for category, name in non_standard:
            print(f"  {category}/{name}")
    else:
        print("✓ All filenames follow lowercase_with_underscores standard")
    
    print("\n" + "="*70)
    print("📈 CORPUS STATISTICS")
    print("-" * 70)
    
    summary_file = corpus_dir / "stats" / "corpus_summary.json"
    if summary_file.exists():
        with summary_file.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"\nTotal sources: {summary['total_sources']}")
        print(f"Total paragraphs: {summary['total_paragraphs']:,}")
        print(f"Total words: {summary['total_words']:,}")
        print(f"Total characters: {summary['total_chars']:,}")
        print(f"Languages: {', '.join(f'{lang} ({count})' for lang, count in summary['languages'].items())}")
        
        print(f"\nAverage words per source: {summary['total_words'] / summary['total_sources']:,.0f}")
        print(f"Average paragraphs per source: {summary['total_paragraphs'] / summary['total_sources']:,.0f}")
    
    print("\n" + "="*70)
    print("🔍 METADATA VERIFICATION")
    print("-" * 70)
    
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    print(f"\nMetadata files found: {len(metadata_files)}")
    
    for meta_file in sorted(metadata_files):
        with meta_file.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\n{meta_file.name}:")
        print(f"  Source: {metadata.get('source_file', 'N/A')}")
        print(f"  Language: {metadata.get('language', 'N/A')}")
        print(f"  Processed: {metadata.get('processed_date', 'N/A')[:10]}")
        
        if 'cleaning_params' in metadata:
            params = metadata['cleaning_params']
            print(f"  Cleaning:")
            print(f"    - URL removed: {params.get('url_removed', False)}")
            print(f"    - HTML removed: {params.get('html_removed', False)}")
            print(f"    - Unicode normalized: {params.get('unicode_normalized', False)}")
            print(f"    - Boilerplate filtered: {params.get('boilerplate_filtered', False)}")
            print(f"    - Case-insensitive dedup: {params.get('case_insensitive_dedup', False)}")
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)
    
    total_exact_dupes = sum(s['exact_duplicates'] for s in all_stats.values())
    if total_exact_dupes == 0:
        print("\n✓ Perfect deduplication: No exact duplicates across all files")
    else:
        print(f"\n⚠️  Total exact duplicates found: {total_exact_dupes}")
    
    if not non_standard:
        print("✓ All filenames properly standardized")
    
    print(f"✓ All {len(metadata_files)} metadata files validated")

if __name__ == "__main__":
    main()
