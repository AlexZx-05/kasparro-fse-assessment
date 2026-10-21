# Kasparro FSE Assessment — AI Claim Extraction and Classification

## Quick start (Windows / Git Bash)

Open a terminal in the project folder and run:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env
notepad .env
```

In `.env`, add your OpenRouter API key and PostgreSQL connection URL:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
DATABASE_URL=your_postgresql_connection_url
```

Keep `.env` private; never paste its contents into chat or commit it to Git.
Langfuse settings are optional. The provided CSV files should be in `data/`.
Check the data and tests:

```bash
python -m app.data_loader
python -m pytest -q
```

Start the API and leave this terminal open:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

When the terminal says `Application startup complete`, open
<http://127.0.0.1:8000/docs> for the interactive API documentation, or visit
<http://127.0.0.1:8000/health> to check that the API is responding. Press
Ctrl+C in the server terminal to stop it.

## Use and check the API

You can use the API in Swagger at <http://127.0.0.1:8000/docs>, or with `curl`
from a second Git Bash terminal. Keep the Uvicorn terminal open while sending
requests.

First, check that the service is responding:

```bash
curl http://127.0.0.1:8000/health
```

The expected response is `{"status":"ok"}`. You can also open
<http://127.0.0.1:8000/docs> in your browser.

To classify an answer, use **POST `/classify`** in Swagger, or send a JSON
request with the `answer_id`, optional `brand_id`, and `ai_answer`. The answer
text must exactly match that answer in `data/answers.csv`; otherwise the API
rejects the request. A successful response contains the extracted claim text,
its start/end character offsets, a label, and supporting catalogue fact IDs.

To start a V1 development evaluation from Git Bash:

```bash
curl -X POST http://127.0.0.1:8000/evaluation-runs \
  -H "Content-Type: application/json" \
  -d '{"version":"v1","dataset_split":"development"}'
```

The response contains a numeric `run_id`, for example:
`{"run_id":4,"version":"v1","dataset_split":"development","status":"running"}`.
For the verified local run 3, open this URL in a browser:

<http://127.0.0.1:8000/evaluation-runs/3>

For any other run, replace `3` in the URL with the numeric `run_id` returned
when you started that run. Poll the GET endpoint to see progress and saved
answer/claim results. `completed` means all answers completed;
`partial` means evaluation finished but some answers had operational or
validation errors; `failed` means the run could not finish. Evaluation runs
execute in the running API process, so do not close or restart Uvicorn until
the run reaches one of these statuses.

### Example of a verified local run

On 21 October 2026, this project's local API completed V1 development run 3
with status `partial`: 37 of 40 answers succeeded and 3 failed exact-source
claim validation. Its measured extraction F1 was 0.9556, joint F1 was 0.9556,
and answer coverage was 0.925. The actual run summary is saved in
`reports/v1_development.json`. This is an example from that run, not a promise
that another run will have the same result.

## How this project works

The project follows a simple evidence-first workflow:

1. It loads the assessment answers and product catalogue from CSV files.
2. The model identifies factual sentences. The application checks that each
   sentence is an exact part of the original answer and calculates its Unicode
   character offsets itself.
3. For classification, the model must call the registered `get_brand_facts`
   tool. The tool returns catalogue records for the answer's brand only.
   Classification is accepted only when labels and cited fact IDs pass
   application validation. A claim can be `supported`, `contradicted`, or
   `unknown`; missing evidence is not treated as contradiction.
4. During development evaluation, the evaluator compares predictions to the
   development gold data after inference and calculates exact-span and
   classification metrics. Gold labels are not sent to the model.
5. The FastAPI service provides classification and evaluation endpoints.
   PostgreSQL stores run and answer results; optional Langfuse observations
   help trace model and tool calls.

Phase 1 establishes the project foundation for loading speaker answers and
catalogue facts, validating the assessment CSVs, and serving a minimal health
endpoint. Phase 2 adds OpenRouter-backed factual claim extraction with source-checked
character offsets. Phase 3 adds catalogue-grounded supported/contradicted/
unknown classification through a native function-tool call. Phase 4 adds a
reproducible development-evaluation harness with exact-span scoring, frozen V1/
V2 configurations, error analysis, and saved metrics/reports.

## Technology stack

- Python 3.10+
- FastAPI and Uvicorn
- Python's standard `csv` module for CSV parsing
- OpenRouter through the official OpenAI Python SDK
- SQLAlchemy and PostgreSQL/Neon for evaluation persistence
- Langfuse Python SDK for optional AI execution traces
- Pydantic for validated extraction and classification results
- `python-dotenv` for local environment configuration
- `pytest` for automated tests

## Project structure

```text
kasparro-fse-assessment/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── data_loader.py
│   ├── main.py
│   └── agent/
│       ├── __init__.py
│       ├── openrouter_client.py
│       ├── claim_extractor.py
│       ├── prompts.py
│       ├── validators.py
│       ├── tools.py
│       └── claim_classifier.py
├── data/
│   ├── answers.csv
│   ├── brand_facts.csv
│   └── development_gold.csv
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_claim_extractor.py
│   └── test_claim_classifier.py
├── reports/
├── prompts/
│   ├── v1_extraction.txt
│   ├── v1_classification.txt
│   ├── v2_extraction.txt
│   └── v2_classification.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- Windows PowerShell
- The three assessment CSV files supplied with the assessment

## Setup (Windows PowerShell)

Run these commands from this project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If script execution is restricted in the current PowerShell session, activate
the environment using the policy approved for your machine, or run
`.venv\Scripts\python.exe` and `.venv\Scripts\pip.exe` directly.

## Dataset placement

Place the supplied files in this project's `data` directory:

- `data\answers.csv`
- `data\brand_facts.csv`
- `data\development_gold.csv`

The originals supplied under the parent `Resource` directory should be copied
into `data`; do not replace or edit those original files. For example, from
this project directory:

```powershell
Copy-Item ..\Resource\answers.csv .\data\answers.csv
Copy-Item ..\Resource\brand_facts.csv .\data\brand_facts.csv
Copy-Item ..\Resource\development_gold.csv .\data\development_gold.csv
```

The loader reads CSV records with UTF-8 decoding and the standard CSV parser,
so quoted commas and embedded newlines are supported. It returns the
`ai_answer` field without trimming or normalizing it.

## Environment configuration

Copy the template to a local `.env` file and edit it locally. Put real
credentials only in `.env`, never in `.env.example`:

```powershell
Copy-Item .env.example .env
notepad .env
```

For Phases 2 and 3, set the OpenRouter key in `.env`. The model defaults to
`openai/gpt-oss-120b` and can be changed with `MODEL_NAME` if needed:

```dotenv
MODEL_NAME=openai/gpt-oss-120b
OPENROUTER_API_KEY=<your OpenRouter API key>
MODEL_TIMEOUT_SECONDS=15
```

The application uses OpenRouter as its fixed provider; it does not switch
providers or infer an endpoint from the model name. The model identifier is
sent as configured to OpenRouter. To switch accounts, replace only
`OPENROUTER_API_KEY`; the key must be created by OpenRouter, not Groq or
another provider.

Configuration errors do not print secret values. Never commit `.env`; it is
excluded by `.gitignore`. If credentials were exposed, revoke/rotate them with
their providers before configuring replacements.

## Validate the CSV files

From the project directory, after placing the data files:

```powershell
python -m app.data_loader
```

The command checks required columns and values, duplicate answer IDs, split
names, CSV record shape, fact value types, development labels, and references
from development gold rows to known answer IDs. It prints only aggregate row
counts, not dataset contents.

The inference-facing `load_inference_data()` returns only answers and product
facts. `load_development_gold()` is a separate function, used by the dataset
validation command; gold labels are not included in the inference data object.

## Run tests

```powershell
python -m pytest -q
```

Tests use temporary synthetic CSV fixtures and do not overwrite the supplied
assessment datasets. Extraction tests mock the provider; no paid API call is
made by the default test run.

## Phase 2 claim extraction

The provider adapter is in `app/agent/openrouter_client.py`; the provider-independent
extraction workflow is in `app/agent/claim_extractor.py`. Extraction rules live
in `app/agent/prompts.py`; exact source-span checking and offset creation are
in `app/agent/validators.py`.

The model returns exact claim text only. Application code locates each claim in
the original answer, verifies the exact substring and sentence boundaries, and
creates Python Unicode code-point offsets and deterministic
`answer_id:start_char:end_char` identifiers. Ambiguous/missing claims raise a
validation error; provider failures are surfaced as sanitized exceptions, not
empty claim lists. Empty or whitespace-only answers return no claims without
calling the provider.

There is no network call at startup, provider fallback, evaluation, database
access, or Langfuse integration in Phase 2.

To run the optional live integration test, first configure a rotated valid key
and a supported model in `.env`, then explicitly enable the test:

```powershell
$env:RUN_OPENROUTER_INTEGRATION = "1"
python -m pytest -q tests\test_claim_extractor.py::test_optional_live_openrouter_integration
Remove-Item Env:RUN_OPENROUTER_INTEGRATION
```

This test makes a real billable request. The normal mocked unit-test suite does
not make one.

## Phase 3 catalogue-grounded classification

`app/agent/tools.py` implements the registered read-only `get_brand_facts`
tool. It loads the real `brand_facts.csv` records through the existing data
loader and returns only facts whose `brand_id` matches the requested brand,
preserving the dataset's actual `fact_id` values. It never loads
`development_gold.csv`.

`app/agent/claim_classifier.py` resolves `answer_id` using `answers.csv`,
checks that the supplied answer exactly matches the assessment record, and
gets the corresponding brand from that row. It sends the extracted claims to
OpenRouter with a native function-tool definition. When the model requests
`get_brand_facts`, Python validates the JSON arguments and requested brand,
executes only that registered function, and appends the actual catalogue result
as a tool message for the final model turn. The first call forces that one
registered function by name; the final request omits tools so it can only
produce the required JSON classification. Unknown tools, another brand,
repeated calls, malformed responses, and exhausted turn limits raise errors.

Model turns mean individual OpenRouter chat-completion requests. The classifier
allows at most three requests per answer, including requests returning tool
calls. A tool request on the final permitted turn fails rather than silently
returning incomplete classifications.

The final model response provides claim IDs, labels, and catalogue fact IDs.
Validation requires every extracted claim exactly once, a valid label, and
fact IDs that were actually returned by the tool. Supported and contradicted
labels must cite evidence. The application rebuilds response claims from the
Phase 2 objects, preserving original claim text, offsets, and deterministic
IDs. Missing evidence is not treated as contradiction; model or catalogue
failures are errors, not `unknown` classifications.

Example local use from the project directory:

```python
from app.agent.claim_classifier import create_default_classifier
from app.agent.claim_extractor import create_default_extractor
from app.data_loader import load_answers

answer = load_answers()[0]
extraction = create_default_extractor().extract(answer.answer_id, answer.ai_answer)
result = create_default_classifier().classify(
    answer.answer_id,
    answer.ai_answer,
    extraction,
)
print(result.model_dump_json(indent=2))
```

The optional live test below performs an actual OpenRouter tool-calling
workflow with the configured model and assessment catalogue. It is separate
from the mocked unit tests.

Enable the live tool-calling integration test in Windows PowerShell only when
`OPENROUTER_API_KEY` is configured in `.env`:

```powershell
$env:RUN_OPENROUTER_TOOL_INTEGRATION = "1"
python -m pytest -q tests\test_claim_classifier.py::test_optional_live_openrouter_tool_calling
Remove-Item Env:RUN_OPENROUTER_TOOL_INTEGRATION
```

This test makes a real request and may incur usage charges. It is skipped by
default and has not been run automatically during ordinary unit-test runs.

## Phase 4 evaluation harness

The project includes a reproducible evaluation runner for the development split.
A frozen Version 1 baseline and a Version 2 candidate are defined by the files
under `prompts/`. Run the evaluator from the project directory with a valid
local `.env` and a model available to your OpenRouter account:

```powershell
python -m app.evaluation.runner --version v1
python -m app.evaluation.runner --version v2
```

Use `--overwrite` only when you intentionally want to replace an existing run's
artifacts. The runner writes JSON metrics, error analysis, and Markdown reports
into `reports/v1/`, `reports/v2/`, and `reports/comparison/` without
consulting development gold during inference.

## Phase 7 final evaluation reports

The frozen prompt versions are identified by SHA-256 hashes in each report.
Run development evaluation through `POST /evaluation-runs` (the API supports
the development split only), then poll `GET /evaluation-runs/{id}` until the
status is `completed` or `partial`. Export the completed PostgreSQL run to its
canonical Phase 7 report without making additional model calls:

```powershell
python -m app.evaluation.runner --version v1 --export-run-id <completed_run_id>
python -m app.evaluation.runner --version v2 --export-run-id <completed_run_id>
```

Run frozen inference on A41-A50 without loading final-check gold labels:

```powershell
python -m app.evaluation.runner --version v1 --split check
python -m app.evaluation.runner --version v2 --split check
```

These commands write `reports/v1_development.json`,
`reports/v2_development.json`, `reports/v1_final.json`, and
`reports/v2_final.json`. Development reports include the existing scorer's
exact-span metrics and confusion matrix. Final-check reports contain actual
inference outcomes but set gold-dependent metrics to `null` when authorized
final labels are unavailable. Runtime is measured locally; actual billed cost
is `null` because the adapter does not expose provider charges. Do not use
`--overwrite` unless replacing that specific run report is intentional.

Metric definitions: extraction precision is exact-span matches divided by
predicted claims; extraction recall is exact-span matches divided by gold
claims. Joint precision and recall require both exact span and correct label.
Macro classification F1 is the unweighted mean of F1 for supported,
contradicted, and unknown. Coverage is successfully processed answers divided
by scheduled answers. Zero-denominator metrics are reported as 0.0. The
confusion matrix includes missing/error and no-gold buckets for exact-span
scoring.

## Phase 5 backend persistence

The project now includes the Phase 5 FastAPI backend layer with:

- `POST /classify` for a single answer
- `POST /evaluation-runs` for an evaluation execution
- `GET /evaluation-runs/{id}` to fetch saved run details

The API uses the existing claim extractor and classifier and stores run,
answer-result, and claim-result records in PostgreSQL through SQLAlchemy.
Startup validates the connection and creates missing application tables
without dropping or recreating existing tables. This is a simple bootstrap,
not a schema migration system; back up the database before any future schema
changes and use migrations for upgrades.

The evaluation API currently accepts the `development` split only. It returns
HTTP 202 with a run ID and `running` status, then executes in a FastAPI
background task. Poll `GET /evaluation-runs/{id}` for completion. This
in-process task is not durable across application restarts; use a durable task
queue before deploying multi-worker production workloads. Evaluation results
now expose the development answer total and successful/failed answer counts as
the run progresses. Each answer's result and claims are persisted immediately
after that answer finishes, while aggregate metrics are saved at the end.
Runs containing failed answers are saved with status `partial`; they
are not presented as fully completed runs. Each API evaluation writes its reports under
`reports/api_runs/<run_id>/<version>/`, preserving the standalone `reports/v1/` and
`reports/v2/` artifacts. Version 2 API runs use the existing standalone V1
metrics as their comparison baseline.

## Start the API

```powershell
uvicorn app.main:app --reload
```

Then check `http://127.0.0.1:8000/health`; it returns `{"status":"ok"}`.

## Current limitations and later phases

Phase 3 includes the catalogue-grounded extraction/classification workflow,
and Phase 4 includes a reproducible evaluation harness for frozen Version 1
and Version 2 runs, exact-span scoring, metric calculation, error analysis,
and report writing. Development gold remains separate from inference and is
never passed to the model. The API's `POST /evaluation-runs` accepts the
development split only and schedules the run as an in-process FastAPI
background task. It does not start final-check evaluation. The task is not
durable across application restarts.

## Phase 6 reliability and observability

- Each OpenRouter request is limited to 15 seconds maximum. `MODEL_TIMEOUT_SECONDS`
  may be set lower, but values above 15 are capped.
- Transient timeouts, connection failures, and provider 5xx errors are retried
  at most once. A temporary HTTP 429 may be retried at most once after a valid
  `Retry-After` delay, only when the delay fits within the remaining timeout.
  Quota-exhaustion errors are not retried. SDK automatic retries are disabled
  so the total attempts stay bounded.
- A process-local semaphore limits active OpenRouter calls to three per application
  process. This is not a cross-worker or deployment-wide limit.
- Langfuse tracing is optional. Set `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`; set
  `LANGFUSE_TRACING_ENABLED=false` to disable it. Missing credentials disable
  tracing without blocking inference.
- Each classified/evaluated answer creates a root observation; OpenRouter
  generations and catalogue tool execution are nested beneath it. Trace
  metadata includes answer ID, split, version, and model. Raw answer text,
  prompts, and catalogue contents are not sent to Langfuse by these helpers.
- Actual prompt/completion token counts are attached only when the provider
  response provides them. Costs are not estimated. The application does not
  currently persist Langfuse trace IDs in PostgreSQL.

Run unit tests from the project directory:

```powershell
python -m pytest -q
```

Start the API:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The live OpenRouter integration, Langfuse ingestion, and Neon persistence have
not been verified as part of this code change. Confirm that the configured
OpenRouter model is enabled for your account before running live requests.
