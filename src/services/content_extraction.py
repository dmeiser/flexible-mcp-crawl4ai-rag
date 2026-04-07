from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_code_blocks(markdown: str) -> List[Dict[str, Any]]:
    """Extract fenced code blocks from markdown text."""
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = []
    for match in pattern.finditer(markdown):
        lang = match.group(1).strip() or None
        code = match.group(2).strip()
        if code:
            blocks.append({"language": lang, "content": code})
    return blocks


def extract_link_references(references_markdown: str) -> List[Dict[str, str]]:
    """Extract structured link references from references markdown text."""
    if not references_markdown or not references_markdown.strip():
        return []

    extracted: List[Dict[str, str]] = []
    for line in references_markdown.splitlines():
        extracted_ref = _extract_reference_from_line(line, len(extracted))
        if extracted_ref is not None:
            extracted.append(extracted_ref)

    return extracted


def _extract_reference_from_line(line: str, existing_count: int) -> Optional[Dict[str, str]]:
    candidate = line.strip()
    if not candidate:
        return None

    for parser in (_parse_markdown_reference, _parse_markdown_link_reference, _parse_plain_url_reference):
        parsed = parser(candidate, existing_count)
        if parsed is not None:
            return parsed
    return None


def _parse_markdown_reference(candidate: str, existing_count: int) -> Optional[Dict[str, str]]:
    _ = existing_count
    markdown_ref = re.match(r"^\[(?P<label>[^\]]+)\]\s*:\s*(?P<url>https?://\S+)(?:\s+(?P<text>.*))?$", candidate)
    if not markdown_ref:
        return None
    return {
        "label": markdown_ref.group("label"),
        "url": markdown_ref.group("url"),
        "text": (markdown_ref.group("text") or "").strip(),
    }


def _parse_markdown_link_reference(candidate: str, existing_count: int) -> Optional[Dict[str, str]]:
    markdown_link = re.search(r"\[(?P<text>[^\]]+)\]\((?P<url>https?://[^)]+)\)", candidate)
    if not markdown_link:
        return None
    return {
        "label": str(existing_count + 1),
        "url": markdown_link.group("url"),
        "text": markdown_link.group("text").strip(),
    }


def _parse_plain_url_reference(candidate: str, existing_count: int) -> Optional[Dict[str, str]]:
    plain_url = re.search(r"(?P<url>https?://\S+)", candidate)
    if not plain_url:
        return None
    label_match = re.match(r"^(?P<label>\d+)[\.:\)]", candidate)
    return {
        "label": label_match.group("label") if label_match else str(existing_count + 1),
        "url": plain_url.group("url"),
        "text": candidate,
    }
