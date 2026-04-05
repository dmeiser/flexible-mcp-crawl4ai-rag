ARG PLAYWRIGHT_BASE_IMAGE=mcr.microsoft.com/playwright/python:v1.58.0-noble

# ---- Builder Stage ----
FROM ${PLAYWRIGHT_BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=true \
    NLTK_DATA=/usr/local/share/nltk_data \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install .

COPY . .
RUN uv pip install .

RUN mkdir -p ${NLTK_DATA} && \
    python -c "import nltk; nltk.download('punkt_tab', download_dir='${NLTK_DATA}', quiet=True)"

# ---- Runtime Stage ----
FROM ${PLAYWRIGHT_BASE_IMAGE} AS runtime

ARG APP_USER=appuser
ARG APP_GROUP=appgroup
ARG UID=1002
ARG GID=1002

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/home/${APP_USER}/.local/bin:${PATH}" \
    PORT=8051 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    EMBEDDING_DIM=768

RUN groupadd -g ${GID} ${APP_GROUP} && \
    useradd -u ${UID} -g ${APP_GROUP} -s /bin/sh -m ${APP_USER}

WORKDIR /app

# Playwright base uses dist-packages (not site-packages) for Python 3.12.
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder ${NLTK_DATA} ${NLTK_DATA}

COPY . .

RUN chown -R ${APP_USER}:${APP_GROUP} /app && \
    mkdir -p /home/${APP_USER}/.cache/ms-playwright ${PLAYWRIGHT_BROWSERS_PATH} && \
    chown -R ${APP_USER}:${APP_GROUP} /home/${APP_USER}/.cache/ms-playwright ${PLAYWRIGHT_BROWSERS_PATH}

USER ${APP_USER}

EXPOSE ${PORT}

CMD ["python", "-m", "src.crawl4ai_mcp"]
