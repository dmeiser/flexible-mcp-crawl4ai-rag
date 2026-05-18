from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import text


def _entity_fields(entity: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (name, etype, desc) from an entity dict."""
    name = entity.get("name", "").strip()
    etype = entity.get("type") or ""
    desc = entity.get("description") or ""
    return name, etype, desc


def _valid_relationship(rel: Any, node_ids: Dict[str, int]) -> Optional[Tuple[int, int, str]]:
    """Return (src_id, tgt_id, relationship) if *rel* is a complete, resolvable edge; else None."""
    if not isinstance(rel, dict):
        return None
    src_name = rel.get("source", "").strip()
    tgt_name = rel.get("target", "").strip()
    relationship = rel.get("relationship", "").strip()
    if not all([src_name, tgt_name, relationship]):
        return None
    src_id = node_ids.get(src_name)
    tgt_id = node_ids.get(tgt_name)
    if src_id is None or tgt_id is None:
        return None
    return src_id, tgt_id, relationship


def _kg_lists(kg_data: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    return kg_data.get("entities") or [], kg_data.get("relationships") or []


async def _store_entities(
    session: Any,
    entities: List[Any],
    source_url: str,
    chunk_id: Optional[int],
    create_embedding_fn: Callable[[str], Any],
) -> Dict[str, int]:
    node_ids: Dict[str, int] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        name, etype, desc = _entity_fields(entity)
        if not name:
            continue
        embed_text = f"{name}: {etype}. {desc}".strip()
        embedding = await create_embedding_fn(embed_text)
        node_id = await _upsert_graph_node(session, entity, source_url, chunk_id, embedding)
        node_ids[name] = node_id
    return node_ids


async def _store_relationships(
    session: Any,
    relationships: List[Any],
    node_ids: Dict[str, int],
    source_url: str,
    chunk_id: Optional[int],
) -> None:
    for rel in relationships:
        validated = _valid_relationship(rel, node_ids)
        if validated is None:
            continue
        src_id, tgt_id, relationship = validated
        await _upsert_graph_edge(session, src_id, tgt_id, relationship, source_url, chunk_id)


async def store_knowledge_graph(
    session: Any,
    kg_data: Dict[str, Any],
    source_url: str,
    chunk_id: Optional[int],
    create_embedding_fn: Callable[[str], Any],
) -> None:
    entities, relationships = _kg_lists(kg_data)
    if not entities and not relationships:
        return
    node_ids = await _store_entities(session, entities, source_url, chunk_id, create_embedding_fn)
    await _store_relationships(session, relationships, node_ids, source_url, chunk_id)
    if node_ids:
        session.commit()


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
