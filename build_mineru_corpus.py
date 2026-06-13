#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Block types to skip in layout.json
SKIP_BLOCK_TYPES = {"ref_text", "image", "header"}

# Pages 0-17: front matter (cover, CIP, committees, TOC, update summary)
# Pages ~155+: references section, empty/blank pages
DEFAULT_PAGE_START = 16
DEFAULT_PAGE_END = 165  # exclusive

# Title keywords that indicate reference/index sections (skip entire page)
REF_SECTION_TITLE_KEYWORDS = ["参考文献", "索引"]

# Line patterns for filtering out reference citation debris
NON_MEDICAL_LINE_PATTERNS = [
    re.compile(r"^\s*\[\d+\]$"),          # standalone "[1]", "[2]"
    re.compile(r"^\s*\{\d+\}$"),          # standalone "{1}"
    re.compile(r"^http[s]?://\S+$"),      # standalone URL
    re.compile(r"^www\.\S+$"),            # standalone www URL
    re.compile(r"^\^{\[.*?\]}$"),         # standalone "^{[xx]}"
    re.compile(r"^\^{.*}$"),             # standalone "^{...}"
]


def _extract_page_blocks(page: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract all (block_type, text) from a page, preserving block order,
    skipping ref_text / image / header types."""
    result: List[Tuple[str, str]] = []
    for block in page.get("para_blocks", page.get("preproc_blocks", [])):
        btype = block.get("type", "")
        if btype in SKIP_BLOCK_TYPES:
            continue
        text = _get_block_text(block)
        if not text:
            continue
        result.append((btype, text))
    return result


def _get_block_text(block: Dict[str, Any]) -> str:
    parts: List[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            content = span.get("content", "")
            if content:
                parts.append(content)
    return " ".join(parts).strip()


def _page_has_ref_section(blocks: List[Tuple[str, str]]) -> bool:
    for btype, text in blocks:
        if btype == "title":
            for kw in REF_SECTION_TITLE_KEYWORDS:
                if kw in text:
                    return True
    return False


def _is_non_medical_line(line: str) -> bool:
    for pat in NON_MEDICAL_LINE_PATTERNS:
        if pat.match(line):
            return True
    return False


def _extract_pages(layout_path: str, page_start: int, page_end: int) -> List[Dict[str, Any]]:
    """Load layout.json and return pages in the given range."""
    with open(layout_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("pdf_info", [])
    print(f"Total pages in layout: {len(pages)}, using range [{page_start}, {page_end})")
    return pages[page_start:page_end]


def build_sections(pages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Group consecutive pages into sections, splitting on title blocks.

    Returns list of (section_title, section_body_text).
    """
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_bodies: List[str] = []

    for page in pages:
        blocks = _extract_page_blocks(page)

        # Drop reference sections within a page instead of skipping the whole page.
        filtered_blocks: List[Tuple[str, str]] = []
        in_ref_section = False
        for btype, text in blocks:
            if btype == "title":
                if any(k in text for k in REF_SECTION_TITLE_KEYWORDS):
                    in_ref_section = True
                    continue
                in_ref_section = False
                filtered_blocks.append((btype, text))
                continue
            if in_ref_section:
                continue
            filtered_blocks.append((btype, text))

        # Merge consecutive titles within this page into a hierarchical title,
        # preserving body text order relative to them.
        merged: List[Tuple[str, str]] = []
        pending_titles: List[str] = []

        for btype, text in filtered_blocks:
            if btype == "title":
                pending_titles.append(text)
            else:
                if pending_titles:
                    merged.append(("title",  " > ".join(pending_titles) if len(pending_titles) > 1 else pending_titles[0]))
                    pending_titles = []
                merged.append((btype, text))

        if pending_titles:
            merged.append(("title",  " > ".join(pending_titles) if len(pending_titles) > 1 else pending_titles[0]))

        # Process merged blocks in order
        for btype, text in merged:
            if btype == "title":
                if current_bodies and current_title:
                    body = "\n".join(current_bodies).strip()
                    if body and len(body) > 20:
                        sections.append((current_title, body))
                current_bodies = []
                current_title = text
            else:
                lines = text.split("\n")
                filtered = [l for l in lines if not _is_non_medical_line(l)]
                if filtered:
                    current_bodies.append("\n".join(filtered))

    # Flush last section
    if current_bodies and current_title:
        body = "\n".join(current_bodies).strip()
        if body and len(body) > 20:
            sections.append((current_title, body))

    return sections


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars <= 0:
        return []
    overlap = max(0, min(overlap_chars, max_chars - 1))
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return out


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    paragraphs = split_paragraphs(text)
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        para = normalize_ws(para)
        if not para:
            continue
        if not buffer:
            if len(para) <= max_chars:
                buffer = para
                continue
            chunks.extend(split_long_text(para, max_chars, overlap_chars))
            continue

        candidate = f"{buffer}\n{para}"
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        chunks.append(buffer.strip())
        buffer = ""
        if len(para) <= max_chars:
            buffer = para
        else:
            chunks.extend(split_long_text(para, max_chars, overlap_chars))

    if buffer:
        chunks.append(buffer.strip())

    return chunks


def build_docs(
    sections: List[Tuple[str, str]],
    dataset_name: str,
    max_chars: int,
    overlap_chars: int,
):
    idx = 0
    for section_title, section_text in sections:
        if not section_text:
            continue

        for chunk in chunk_text(section_text, max_chars=max_chars, overlap_chars=overlap_chars):
            if len(chunk) < 10:
                continue
            idx += 1
            doc_id = f"{dataset_name}_{idx:06d}"
            yield {
                "_id": doc_id,
                "title": section_title or dataset_name,
                "text": chunk,
            }


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {
                "id" if "chunk" in str(path) else "_id": row.get("_id"),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build searchable corpus from MinerU layout + markdown")
    parser.add_argument("--layout-json", required=True, help="Path to MinerU layout.json")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--chunk-chars", type=int, default=600, help="Max chars per chunk")
    parser.add_argument("--overlap-chars", type=int, default=80, help="Overlap chars between chunks")
    parser.add_argument("--page-start", type=int, default=DEFAULT_PAGE_START, help="First layout page to include")
    parser.add_argument("--page-end", type=int, default=DEFAULT_PAGE_END, help="Last layout page to include (exclusive)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    dataset_name = out_dir.name

    pages = _extract_pages(args.layout_json, args.page_start, args.page_end)

    print(f"Building sections from {len(pages)} pages...")
    sections = build_sections(pages)
    print(f"Found {len(sections)} sections")

    docs = list(
        build_docs(
            sections=sections,
            dataset_name=dataset_name,
            max_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
        )
    )

    corpus_path = out_dir / "corpus.jsonl"
    chunk_path = out_dir / "chunk" / f"{dataset_name}.jsonl"

    write_jsonl(corpus_path, docs)
    write_jsonl(chunk_path, docs)

    print(f"Wrote {len(docs)} docs to {corpus_path}")
    print(f"Wrote chunk shards to {chunk_path}")


if __name__ == "__main__":
    main()
