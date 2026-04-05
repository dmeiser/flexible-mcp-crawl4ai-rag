"""Functions for extracting metadata from crawled content."""
import re
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }


def extract_link_graph(markdown: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Extract lightweight link graph information from markdown text."""
    if not markdown:
        return {
            "total_links": 0,
            "internal_links": 0,
            "external_links": 0,
            "links": [],
        }

    base_host = urlparse(base_url).netloc if isinstance(base_url, str) and base_url else ""
    matches = re.findall(r"\[(?P<text>[^\]]*)\]\((?P<url>[^)]+)\)", markdown)

    links: List[Dict[str, Any]] = []
    internal = 0
    external = 0

    for text, href in matches:
        href = href.strip()
        if not href:
            continue
        parsed = urlparse(href)
        host = parsed.netloc
        is_external = bool(host and base_host and host != base_host)
        is_internal = not is_external
        if is_internal:
            internal += 1
        else:
            external += 1
        links.append(
            {
                "url": href,
                "anchor_text": text.strip(),
                "is_external": is_external,
            }
        )

    return {
        "total_links": len(links),
        "internal_links": internal,
        "external_links": external,
        "links": links,
    }


def extract_media_metadata(markdown: str) -> Dict[str, Any]:
    """Extract image/video/audio style references from markdown text."""
    if not markdown:
        return {
            "images": [],
            "media_links": [],
            "image_count": 0,
            "media_link_count": 0,
        }

    image_matches = re.findall(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)", markdown)
    images = [
        {
            "url": url.strip(),
            "alt_text": alt.strip(),
        }
        for alt, url in image_matches
        if isinstance(url, str) and url.strip()
    ]

    media_matches = re.findall(r"\[(?P<text>[^\]]*)\]\((?P<url>[^)]+)\)", markdown)
    media_links: List[Dict[str, str]] = []
    for text, url in media_matches:
        normalized = url.strip().lower()
        if not normalized:
            continue
        if normalized.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".webm", ".mp3", ".wav", ".pdf")):
            media_links.append({"url": url.strip(), "text": text.strip()})

    return {
        "images": images,
        "media_links": media_links,
        "image_count": len(images),
        "media_link_count": len(media_links),
    }