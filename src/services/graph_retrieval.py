from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy import text


def _query_seed_nodes(
    session: Any,
    emb_str: str,
    entity_type_filter: Optional[str],
    match_count: int,
) -> list:
    return session.execute(
        text(
            """
            SELECT id, entity_name, entity_type, description, source_url,
                   1 - (embedding <=> :emb::vector) AS similarity
            FROM graph_nodes
            WHERE embedding IS NOT NULL
              AND (:type_filter IS NULL OR entity_type = :type_filter)
            ORDER BY embedding <=> :emb::vector
            LIMIT :cnt
            """
        ),
        {"emb": emb_str, "type_filter": entity_type_filter, "cnt": match_count},
    ).fetchall()


def _build_seed_data(seed_rows: list) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    seed_ids = [int(row[0]) for row in seed_rows]
    seed_map: Dict[int, Dict[str, Any]] = {
        int(row[0]): {
            "entity_name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "source_url": row[4],
            "similarity": float(row[5]),
        }
        for row in seed_rows
    }
    return seed_ids, seed_map


def _query_neighborhood(session: Any, seed_ids: List[int], depth: int) -> Dict[int, List[str]]:
    rows = session.execute(
        text(
            """
            WITH RECURSIVE neighborhood AS (
                SELECT source_node_id, target_node_id, relationship, 1 AS hop_depth,
                       ARRAY[source_node_id, target_node_id] AS visited,
                       ARRAY[relationship] AS rel_chain
                FROM graph_edges WHERE source_node_id = ANY(:seed_ids)
                UNION ALL
                SELECT ge.source_node_id, ge.target_node_id, ge.relationship, n.hop_depth + 1,
                       n.visited || ge.target_node_id,
                       n.rel_chain || ge.relationship
                FROM graph_edges ge
                JOIN neighborhood n ON ge.source_node_id = n.target_node_id
                WHERE n.hop_depth < :max_depth
                  AND NOT ge.target_node_id = ANY(n.visited)
            )
            SELECT DISTINCT target_node_id, rel_chain FROM neighborhood
            """
        ),
        {"seed_ids": seed_ids, "max_depth": depth},
    ).fetchall()
    neighbor_map: Dict[int, List[str]] = {}
    for row in rows:
        nid = int(row[0])
        if nid not in neighbor_map:
            neighbor_map[nid] = list(row[1]) if row[1] else []
    return neighbor_map


def _query_node_info(session: Any, node_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT id, entity_name, entity_type, description, source_url
            FROM graph_nodes
            WHERE id = ANY(:ids)
            """
        ),
        {"ids": node_ids},
    ).fetchall()
    return {
        int(row[0]): {
            "entity_name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "source_url": row[4],
        }
        for row in rows
    }


def _query_url_content(session: Any, urls: List[str]) -> Dict[str, str]:
    rows = session.execute(
        text(
            """
            SELECT url, content FROM crawled_pages
            WHERE url = ANY(:urls) AND is_active = TRUE AND tombstoned_at IS NULL
            ORDER BY id
            LIMIT 1000
            """
        ),
        {"urls": urls},
    ).fetchall()
    url_content_map: Dict[str, str] = {}
    for row in rows:
        if row[0] not in url_content_map:
            url_content_map[row[0]] = row[1]
    return url_content_map


def _build_node_result(
    node_id: int,
    seen: Set[int],
    node_info: Dict[int, Dict[str, Any]],
    url_content_map: Dict[str, str],
    rel_chain: List[str],
    similarity: float,
) -> Optional[Dict[str, Any]]:
    """Return a result dict for *node_id* if unseen and resolvable, else None."""
    if node_id in seen:
        return None
    seen.add(node_id)
    info = node_info.get(node_id)
    if not info:
        return None
    source_url = info["source_url"]
    return {
        "url": source_url,
        "content": url_content_map.get(source_url, ""),
        "entity_name": info["entity_name"],
        "entity_type": info["entity_type"],
        "relationship_chain": rel_chain,
        "graph_context": f"{info['entity_name']} ({info['entity_type']}): {info['description']}",
        "similarity_score": similarity,
    }


def _build_results(
    seed_ids: List[int],
    seed_map: Dict[int, Dict[str, Any]],
    neighbor_map: Dict[int, List[str]],
    node_info: Dict[int, Dict[str, Any]],
    url_content_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for node_id in seed_ids:
        r = _build_node_result(node_id, seen, node_info, url_content_map, [], seed_map[node_id]["similarity"])
        if r is not None:
            results.append(r)
    for node_id, rel_chain in neighbor_map.items():
        r = _build_node_result(node_id, seen, node_info, url_content_map, rel_chain, 0.0)
        if r is not None:
            results.append(r)
    return results


async def search_knowledge_graph(
    session: Any,
    query: str,
    create_embedding_fn: Callable[[str], Any],
    match_count: int = 5,
    depth: int = 2,
    entity_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_embedding = await create_embedding_fn(query)
    seed_rows = _query_seed_nodes(session, str(query_embedding), entity_type_filter, match_count)
    if not seed_rows:
        return []
    seed_ids, seed_map = _build_seed_data(seed_rows)
    neighbor_map = _query_neighborhood(session, seed_ids, depth)
    all_node_ids = list(set(seed_ids) | set(neighbor_map.keys()))
    node_info = _query_node_info(session, all_node_ids)
    all_source_urls = list({info["source_url"] for info in node_info.values()})
    url_content_map = _query_url_content(session, all_source_urls)
    return _build_results(seed_ids, seed_map, neighbor_map, node_info, url_content_map)
