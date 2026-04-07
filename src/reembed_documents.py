"""
Utility script to re-embed all stored documents with the current embedding model.
Run with: uv run python -m src.reembed_documents
"""

import asyncio
import logging

from sqlmodel import select

from .utils import CodeExample, CrawledPage, create_embedding, get_session, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reembed_all() -> None:
    logger.info(f"Re-embedding with provider={settings.EMBEDDING_PROVIDER}, model={_current_model()}")

    with next(get_session()) as session:
        pages = session.exec(select(CrawledPage)).all()
        logger.info(f"Found {len(pages)} crawled page chunks to re-embed.")
        for page in pages:
            try:
                page.embedding = await create_embedding(page.content)
                session.add(page)
            except Exception as exc:
                logger.error(f"Failed to re-embed page id={page.id}: {exc}")
        session.commit()
        logger.info("Crawled pages re-embedded.")

        examples = session.exec(select(CodeExample)).all()
        logger.info(f"Found {len(examples)} code examples to re-embed.")
        for ex in examples:
            try:
                ex.embedding = await create_embedding(ex.content)
                session.add(ex)
            except Exception as exc:
                logger.error(f"Failed to re-embed code example id={ex.id}: {exc}")
        session.commit()
        logger.info("Code examples re-embedded.")


def _current_model() -> str:
    return settings.effective_embedding_model_name


if __name__ == "__main__":
    asyncio.run(reembed_all())
