#!/usr/bin/env python3

import csv
import re
from pathlib import Path
from typing import List


def normalize_and_split(text: str) -> List[str]:
	if text is None:
		return []
	t = text.replace('\r\n', '\n').replace('\r', '\n')
	t = t.strip().strip('"')
	if not t:
		return []
	t = re.sub(r"\n{3,}", "\n\n", t)
	parts = re.split(r"\n\s*\n", t)
	cleaned = []
	for p in parts:
		p = p.strip()
		if not p:
			continue
		p = re.sub(r"[ \t]{2,}", " ", p)
		cleaned.append(p)
	return cleaned


def detect_text_column(header, sample_rows):
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
			if val is None:
				continue
			s = str(val)
			sums[i] += len(s)
			counts[i] += 1
	avg = [(sums[i] / counts[i]) if counts[i] else 0 for i in range(col_count)]
	best = max(range(col_count), key=lambda i: avg[i])
	return best


def process_csv(inpath: Path, outpath: Path):
	paragraphs = []
	sample = []
	header = None
	with inpath.open("r", encoding="utf-8", newline='') as f:
		reader = csv.reader(f)
		try:
			header = next(reader)
		except StopIteration:
			header = None
		for _ in range(200):
			try:
				row = next(reader)
			except StopIteration:
				break
			sample.append(row)

	text_idx = detect_text_column(header, sample)

	with inpath.open("r", encoding="utf-8", newline='') as f:
		reader = csv.reader(f)
		first_row = None
		try:
			first_row = next(reader)
		except StopIteration:
			first_row = None

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
			if text_idx >= len(row):
				candidate = " ".join(row)
			else:
				candidate = row[text_idx]
			if candidate is None:
				continue
			candidate = str(candidate)
			parts = normalize_and_split(candidate)
			paragraphs.extend(parts)

	seen = set()
	deduped = []
	for p in paragraphs:
		if p in seen:
			continue
		seen.add(p)
		deduped.append(p)

	outpath.parent.mkdir(parents=True, exist_ok=True)
	with outpath.open("w", encoding="utf-8") as f:
		for i, p in enumerate(deduped):
			f.write(p)
			if i != len(deduped) - 1:
				f.write("\n\n")

	return len(deduped)


def main():
	workspace = Path(__file__).resolve().parent
	inputs = sorted(workspace.glob("*.csv"))
	results = {}
	for inp in inputs:
		if not inp.exists():
			print(f"Skipping missing file: {inp}")
			continue
		out = inp.with_suffix("").with_name(inp.stem + ".txt")
		count = process_csv(inp, out)
		results[inp.name] = (out, count)

	print("Done. Summary:")
	for k, (outp, cnt) in results.items():
		print(f"  {k} -> {outp.name}: {cnt} paragraphs")


if __name__ == "__main__":
	main()

