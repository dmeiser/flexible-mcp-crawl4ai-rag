# LLM Tool Test Prompt

Paste this whole document into the LLM that will run the tool checks.

---

You are testing every exposed MCP tool in this project.

Your job is to:

1. call each tool,
2. validate the expected result,
3. persist a test record,
4. verify the test record was persisted,
5. continue even if a tool fails,
6. produce a final pass/fail/skipped summary.

## Non-negotiable rules

- Do not mark a tool as PASS unless both the tool assertion and persistence verification succeed.
- Do not stop on the first failure.
- Record every outcome: PASS, FAIL, or SKIPPED.
- If a tool is optional and unavailable, mark it SKIPPED with a reason.
- For expected error-path tests, mark PASS only if the tool fails in the expected way.

## Persistence requirement

For every tool test, store a ledger entry.

### Primary storage method

Use `index_markdown` to store a ledger record at a unique URL:

- `https://llm-tool-test.internal/{run_id}/{tool_name}`

Then verify that record with `get_markdown_by_url`.

### Optional SQL audit trail

If a SQL/database tool is available, also insert a row into a table like:

- `llm_tool_test_results(run_id, tool_name, status, assertion_summary, response_json, created_at)`

If SQL is not available, do not block on it. The `index_markdown` ledger is required; SQL is optional.

## Values to create at the start

Create these values before testing:

- `run_id`: unique string, for example `2026-04-05T12-34-56Z`
- `test_url = https://example.com`
- `smoke_base_url = https://smoke-test.internal`
- `index_url = {smoke_base_url}/index-markdown-test-{run_id}`
- `fit_url = {smoke_base_url}/index-fit-markdown-test-{run_id}`
- `struct_url = {smoke_base_url}/index-structured-test-{run_id}`
- `code_url = {smoke_base_url}/index-code-test-{run_id}`
- `provenance_url = {smoke_base_url}/provenance-test-{run_id}`

## Reusable fixtures

### Markdown fixture

```text
# Smoke Test Document

This is e2e smoke test content for retrieval verification.
```

### Fit markdown fixture

```text
# Fit Markdown Smoke Test

Indexed fit markdown for retrieval testing.
```

### Structured content fixture

```json
{"title":"Smoke Test Record","tags":["e2e","structured"]}
```

### Code markdown fixture

~~~text
# Code Example

```python
def hello_world():
    return 'Hello, World!'
```
~~~

### Provenance metadata

```json
{
  "smoke_test": true,
  "markdown_variant": "raw_markdown",
  "references_markdown": "[1]: https://example.com/reference Example reference",
  "has_citations": true
}
```

## Ledger entry format

After each tool test, store a record with `index_markdown` using this structure:

```text
# Tool Test Result: {tool_name}

- run_id: {run_id}
- tool_name: {tool_name}
- status: PASS | FAIL | SKIPPED
- assertion_summary: {short summary}

## Request
{request_json}

## Response
{response_json}
```

Use metadata like:

```json
{"test_run":true,"run_id":"{run_id}","tool_name":"{tool_name}","status":"PASS"}
```

Verification after storing:

- call `get_markdown_by_url` on the ledger URL
- assert `success == true`
- assert `chunk_count >= 1`
- assert returned content contains the `tool_name`

## Execution loop

For each tool below:

1. call the tool,
2. validate the listed assertions,
3. store a ledger record with `index_markdown`,
4. verify the ledger record with `get_markdown_by_url`,
5. if available, also store SQL evidence,
6. record PASS, FAIL, or SKIPPED,
7. continue.

If the tool fails unexpectedly, still store a FAIL ledger record.

---

## 1. Tool registration

If your client can list tools, verify these are present:

- `crawl_url`
- `crawl_to_markdown`
- `crawl_many_urls`
- `crawl_local_file`
- `crawl_raw_html`
- `crawl_deep`
- `crawl_adaptive`
- `crawl_with_session`
- `crawl_with_auth_hooks`
- `crawl_login_required`
- `crawl_paginated`
- `crawl_with_browser_config`
- `ingest_content_directory`
- `create_session`
- `inspect_session`
- `kill_session`
- `extract_markdown_variants`
- `extract_fit_markdown`
- `extract_structured_json`
- `generate_extraction_schema`
- `validate_extraction_schema`
- `extract_regex_entities`
- `extract_knowledge_graph`
- `extract_code_examples`
- `index_markdown`
- `index_fit_markdown`
- `index_structured_content`
- `index_code_examples`
- `search_documents`
- `search_raw_markdown`
- `search_fit_markdown`
- `search_structured_content`
- `get_document_by_id`
- `get_markdown_by_url`
- `get_fit_markdown_by_url`

Optional:

- `search_code_examples`

Also verify these are not exposed:

- `compute_value_scores`
- `preview_eviction_plan`
- `enforce_storage_budget`
- `pin_records`
- `unpin_records`
- `index_storage_report`
- `restore_tombstoned_records`
- `recrawl_due_sources`
- `prune_stale_content`
- `hard_delete_tombstones`
- `detect_content_drift`
- `crawl_single_page`
- `smart_crawl_url`
- `get_available_sources`
- `perform_rag_query`
- `search_documents_tool`

Store result under:

- `https://llm-tool-test.internal/{run_id}/tool_registration`

---

## 2. Crawl and write-path tools

### `crawl_url`

Request:

```json
{"url":"https://example.com","mode":"markdown","markdown_variant":"raw","index_result":true}
```

Assert:

- `success == true`
- `chunks_stored >= 1`

### `crawl_deep`

Assert:

- `success == true`
- `pages_crawled >= 1`
- `chunks_stored >= 1`

### `crawl_adaptive`

Run twice:

1. normal adaptive crawl
2. adaptive crawl with export + answer

Assert on normal run:

- `success == true`
- `pages_crawled >= 1`
- `chunks_stored >= 1`

Assert on export run:

- `success == true`
- `knowledge_base_export.format in {"json", "jsonl"}`
- `adaptive_answer.query` matches your prompt

### `crawl_to_markdown`

Run twice:

1. standard call
2. call with `index_variants: "both"`

Assert:

- `success == true`
- `pages_crawled >= 1`
- `chunks_stored >= 1`
- second run has `index_variants_override == "both"`
- `indexed_variants` contains `raw_markdown`

### `crawl_many_urls`

Assert:

- `success == true`
- `pages_crawled >= 1`
- `chunks_stored >= 1`

### `crawl_raw_html`

Use HTML:

```html
<html><body><h1>Raw HTML Smoke Test</h1><p>Content for e2e testing.</p></body></html>
```

Assert:

- `success == true`
- `selected_markdown` is non-empty
- `chunks_stored >= 1`

### `crawl_local_file`

Use `/app/pyproject.toml`

Assert:

- `success == true`
- `chunks_stored >= 1`

### `create_session`, `inspect_session`, `kill_session`

Use session id:

- `llm-tool-test-session-{run_id}`

Assert:

- create succeeds
- inspect succeeds
- inspect says `active == true`
- kill succeeds

### `crawl_with_session`

Assert:

- `success == true`
- `chunks_stored >= 1`

### `crawl_with_browser_config`

Use browser config:

```json
{"text_mode":true}
```

Assert:

- `success == true`
- `chunks_stored >= 1`

### `crawl_with_auth_hooks`

Assert:

- `success == true`
- `workflow_mode == "direct_auth_hooks"`
- `chunks_stored >= 1`

### `crawl_login_required`

Assert:

- `success == true`
- `chunks_stored >= 1`

### `crawl_paginated`

Assert:

- `success == true`
- `chunks_stored >= 1`

### `ingest_content_directory`

Use:

```json
{"directory_path":"/app","include_patterns":["pyproject.toml","README.md"],"index_result":true}
```

Assert:

- `success == true`
- `files_discovered >= 1`
- `indexed_count >= 1`

---

## 3. Extraction tools

### `extract_markdown_variants`

Run twice:

1. `index_result: false`
2. `index_result: true, index_variants: "both"`

Assert:

- `success == true`
- response contains `raw_markdown`
- response contains `fit_markdown`
- indexed run has `chunks_stored >= 1`

### `extract_fit_markdown`

Assert:

- `success == true`
- `selected_variant == "fit_markdown"`

### `extract_structured_json`

Assert:

- `success == true`

### `generate_extraction_schema`

Assert:

- `success == true`
- response contains `schema`

### `validate_extraction_schema`

Assert:

- `success == true`

### `extract_regex_entities`

Use pattern:

```json
{"links":"https?://[^\\s\\\"']+"}
```

Assert:

- `success == true`

### `extract_code_examples`

Assert:

- `success == true`
- response contains `code_examples`

### `extract_knowledge_graph`

If available and configured:

- assert `success == true`

If unavailable or intentionally disabled:

- mark `SKIPPED`
- store the reason in the ledger

---

## 4. Indexing tools

### `index_markdown`

Use `index_url` and the markdown fixture.

Assert:

- `success == true`
- `chunks_stored >= 1`
- `first_chunk_id` exists
- `first_chunk_id > 0`

Save `first_chunk_id` for the later document round-trip test.

### `index_fit_markdown`

Use `fit_url`.

Assert:

- `success == true`
- `chunks_stored >= 1`

### `index_structured_content`

Use `struct_url`.

Assert:

- `success == true`
- `chunks_stored >= 1`

### `index_code_examples`

Use `code_url`.

Assert:

- `success == true`
- `code_examples_indexed >= 1`

### provenance `index_markdown`

Use `provenance_url` with the provenance metadata.

Assert:

- `success == true`
- `chunks_stored >= 1`

### chunking-strategy `index_markdown`

Run three subtests with:

- `chunking_strategy: "sentence"`
- `chunking_strategy: "fixed"`
- `chunking_strategy: "paragraph"`

Assert for each:

- `success == true`
- `chunks_stored >= 1`
- `chunking_strategy_applied` equals the requested strategy

---

## 5. Retrieval tools

Run retrieval only after indexing is complete.

### `search_documents`

Run twice with:

1. `what is example.com about`
2. `smoke test retrieval verification`

Assert:

- `success == true`
- `len(results) >= 1`
- first run top content includes `example`

### `search_raw_markdown`

Assert:

- `success == true`
- `len(results) >= 1`

### `search_fit_markdown`

Assert:

- `success == true`
- `len(results) >= 1`

### `search_structured_content`

Assert:

- `success == true`
- `len(results) >= 1`

### `search_code_examples`

If exposed:

- `success == true`
- `len(results) >= 1`

If not exposed:

- mark `SKIPPED`
- store reason `tool not registered`

### `search_documents` with provenance

Use query:

- `proves provenance retrieval works`

Set:

- `include_provenance: true`

Assert:

- `success == true`
- one result has `url == provenance_url`
- `provenance.has_citations == true`
- first provenance reference URL is `https://example.com/reference`

### `search_documents` with freshness controls

Set:

```json
{"fresh_only":true,"as_of":"2999-01-01T00:00:00+00:00","recency_bias":0.4}
```

Assert:

- `success == true`
- `fresh_only == true`
- `as_of` is not null
- if results exist, top result includes `final_score` and `freshness_score`

### `get_document_by_id`

Use the saved `first_chunk_id`.

Assert:

- `success == true`
- response contains `document`
- `document.id == first_chunk_id`
- `document.url == index_url`

### `get_markdown_by_url`

Use previously indexed URLs and ledger URLs.

Assert:

- `success == true`
- `chunk_count >= 1`

### `get_fit_markdown_by_url`

Use `fit_url`.

Assert:

- `success == true`
- `chunk_count >= 1`

### `get_markdown_by_url` with provenance

Use `provenance_url` and `include_provenance: true`.

Assert:

- `success == true`
- `provenance.has_citations == true`
- first provenance reference URL is `https://example.com/reference`

---

## 6. Error-path tests

These count as PASS only if the expected failure behavior occurs.

### `index_markdown` with empty markdown

Assert:

- `success == false`
- response contains `error`

### `crawl_raw_html` with empty html

Assert:

- `success == false`
- response contains `error`

### `get_document_by_id` with nonexistent id

Use `999999999`.

Assert:

- `success == false`
- response contains `error`

---

## Final output format

At the end, output a table with these columns:

- `tool_name`
- `status`
- `primary_assertion`
- `ledger_url`
- `ledger_verified`
- `notes`

Then output totals:

- total tested
- passed
- failed
- skipped

A complete success means:

- all required tools passed,
- every PASS has a verified ledger record,
- optional tools are PASS or SKIPPED with a reason,
- all error-path tests behaved as expected.
