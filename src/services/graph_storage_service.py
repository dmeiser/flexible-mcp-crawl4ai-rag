from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import text


async def store_knowledge_graph(
    session: Any,
    kg_data: Dict[str, Any],
    source_url: str,
    chunk_id: Optional[int],
    create_embedding_fn: Callable[[str], Any],
) -> None:
    entities: List[Dict[str, Any]] = kg_data.get("entities") or []
    relationships: List[Dict[str, Any]] = kg_data.get("relationships") or []
    if not entities and not relationships:
        return

    node_ids: Dict[str, int] = {}
    for entity in entities:
        name = entity.get("name", "").strip()
        if not name:
            continue
        etype = entity.get("type") or ""
        desc = entity.get("description") or ""
        embed_text = f"{name}: {etype}. {desc}".strip()
        embedding = await create_embedding_fn(embed_text)
        node_id = await _upsert_graph_node(session, entity, source_url, chunk_id, embedding)
        node_ids[name] = node_id

    for rel in relationships:
        src_name = rel.get("source", "").strip()
        tgt_name = rel.get("target", "").strip()
        relationship = rel.get("relationship", "").strip()
        if not src_name or not tgt_name or not relationship:
            continue
        src_id = node_ids.get(src_name)
        tgt_id = node_ids.get(tgt_name)
        if src_id is None or tgt_id is None:
            continue
        await _upsert_graph_edge(session, src_id, tgt_id, relationship, source_url, chunk_id)


async def _upsert_graph_node(
    session: Any,
    entity: Dict[str, Any],
    source_url: str,
    chunk_id: Optional[int],
    embedding: List[float],
) -> int:
    name = entity.get("name", "").strip()
    etype = entity.get("type") or None
    desc = entity.get("description") or None
    result = session.execute(
        text(
            """
            INSERT INTO graph_nodes (entity_name, entity_type, source_url, chunk_id, description, embedding)
            VALUES (:name, :etype, :source_url, :chunk_id, :desc, CAST(:emb AS vector))
            ON CONFLICT (entity_name, source_url) DO UPDATE
                SET entity_type = EXCLUDED.entity_type,
                    description  = EXCLUDED.description,
                    embedding    = EXCLUDED.embedding
            RETURNING id
            """
        ),
        {
            "name": name,
            "etype": etype,
            "source_url": source_url,
            "chunk_id": chunk_id,
            "desc": desc,
            "emb": str(embedding),
        },
    )
    row = result.fetchone()
    return int(row[0])


async def _upsert_graph_edge(
    session: Any,
    source_node_id: int,
    target_node_id: int,
    relationship: str,
    source_url: str,
    chunk_id: Optional[int],
) -> None:
    session.execute(
        text(
            """
            INSERT INTO graph_edges (source_node_id, target_node_id, relationship, source_url, chunk_id)
            VALUES (:src, :tgt, :rel, :source_url, :chunk_id)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "src": source_node_id,
            "tgt": target_node_id,
            "rel": relationship,
            "source_url": source_url,
            "chunk_id": chunk_id,
        },
    )
