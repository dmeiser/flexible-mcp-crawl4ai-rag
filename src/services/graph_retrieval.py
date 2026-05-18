from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import text


async def search_knowledge_graph(
    session: Any,
    query: str,
    create_embedding_fn: Callable[[str], Any],
    match_count: int = 5,
    depth: int = 2,
    entity_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_embedding = await create_embedding_fn(query)
    emb_str = str(query_embedding)

    seed_rows = session.execute(
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

    if not seed_rows:
        return []

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

    neighborhood_rows = session.execute(
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
    for row in neighborhood_rows:
        nid = int(row[0])
        rel_chain: List[str] = list(row[1]) if row[1] else []
        if nid not in neighbor_map:
            neighbor_map[nid] = rel_chain

    all_node_ids = list(set(seed_ids) | set(neighbor_map.keys()))

    node_rows = session.execute(
        text(
            """
            SELECT id, entity_name, entity_type, description, source_url
            FROM graph_nodes
            WHERE id = ANY(:ids)
            """
        ),
        {"ids": all_node_ids},
    ).fetchall()

    node_info: Dict[int, Dict[str, Any]] = {}
    for row in node_rows:
        node_info[int(row[0])] = {
            "entity_name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "source_url": row[4],
        }

    all_source_urls = list({info["source_url"] for info in node_info.values()})
    content_rows = session.execute(
        text(
            """
            SELECT url, content FROM crawled_pages
            WHERE url = ANY(:urls) AND is_active = TRUE AND tombstoned_at IS NULL
            ORDER BY id
            LIMIT 1000
            """
        ),
        {"urls": all_source_urls},
    ).fetchall()

    url_content_map: Dict[str, str] = {}
    for row in content_rows:
        if row[0] not in url_content_map:
            url_content_map[row[0]] = row[1]

    results: List[Dict[str, Any]] = []
    seen_node_ids: set = set()
    for node_id in seed_ids:
        if node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)
        info = node_info.get(node_id)
        if not info:
            continue
        source_url = info["source_url"]
        results.append(
            {
                "url": source_url,
                "content": url_content_map.get(source_url, ""),
                "entity_name": info["entity_name"],
                "entity_type": info["entity_type"],
                "relationship_chain": [],
                "graph_context": f"{info['entity_name']} ({info['entity_type']}): {info['description']}",
                "similarity_score": seed_map[node_id]["similarity"],
            }
        )

    for node_id, rel_chain in neighbor_map.items():
        if node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)
        info = node_info.get(node_id)
        if not info:
            continue
        source_url = info["source_url"]
        results.append(
            {
                "url": source_url,
                "content": url_content_map.get(source_url, ""),
                "entity_name": info["entity_name"],
                "entity_type": info["entity_type"],
                "relationship_chain": rel_chain,
                "graph_context": f"{info['entity_name']} ({info['entity_type']}): {info['description']}",
                "similarity_score": 0.0,
            }
        )

    return results
