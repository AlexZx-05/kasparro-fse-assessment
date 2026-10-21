# Kasparro FSE Assessment — AI Claim Extraction and Classification

Phase 1 establishes the project foundation for loading speaker answers and
catalogue facts, validating the assessment CSVs, and serving a minimal health
endpoint. Phase 2 adds Groq-backed factual claim extraction with source-checked
character offsets. Phase 3 adds catalogue-grounded supported/contradicted/
unknown classification through a native Groq function-tool call. Phase 4 adds a
reproducible development-evaluation harness with exact-span scoring, frozen V1/
V2 configurations, error analysis, and saved metrics/reports.

## Technology stack

- Python 3.10+
- FastAPI and Uvicorn
- Python's standard `csv` module for CSV parsing
- Groq's official Python SDK for model requests
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
│       ├── groq_client.py
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
│   ├── v1.txt
│   └── v2.txt
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

For Phases 2 and 3, set these values in `.env`:

```dotenv
MODEL_PROVIDER=groq
MODEL_NAME=openai/gpt-oss-120b
GROQ_API_KEY=<your rotated Groq API key>
GROQ_TIMEOUT_SECONDS=30
```

Groq announced shutdown of `llama-3.3-70b-versatile` on August 16, 2026 for
free and developer tiers. `openai/gpt-oss-120b` is one of Groq's documented
replacement models; confirm it is active for your account in Groq Console.

Database and Langfuse variables remain placeholders and are unused in Phases 2
and 3.
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

The provider adapter is in `app/agent/groq_client.py`; the provider-independent
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
$env:RUN_GROQ_INTEGRATION = "1"
python -m pytest -q tests\test_claim_extractor.py::test_optional_live_groq_integration
Remove-Item Env:RUN_GROQ_INTEGRATION
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
Groq with a native function-tool definition. When Groq requests
`get_brand_facts`, Python validates the JSON arguments and requested brand,
executes only that registered function, and appends the actual catalogue result
as a tool message for the final model turn. The first call forces that one
registered function by name; the final request omits tools so it can only
produce the required JSON classification. Unknown tools, another brand,
repeated calls, malformed responses, and exhausted turn limits raise errors.

Model turns mean individual Groq chat-completion requests. The classifier
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

Groq's [tool-use documentation](https://console.groq.com/docs/tool-use) states
that hosted models support tool use. The optional live test below performs an
actual Groq tool-calling workflow with the configured model and assessment
catalogue. It is separate from the mocked unit tests.

Enable the live tool-calling integration test in Windows PowerShell only when
the rotated `GROQ_API_KEY` is configured in `.env`:

```powershell
$env:RUN_GROQ_TOOL_INTEGRATION = "1"
python -m pytest -q tests\test_claim_classifier.py::test_optional_live_groq_tool_calling
Remove-Item Env:RUN_GROQ_TOOL_INTEGRATION
```

This test makes a real request and may incur usage charges. It is skipped by
default and has not been run automatically during ordinary unit-test runs.

## Phase 4 evaluation harness

The project includes a reproducible evaluation runner for the development split.
A frozen Version 1 baseline and a Version 2 candidate are defined by the files
under `prompts/`. Run the evaluator from the project directory with a valid
local `.env` and a supported Groq model:

```powershell
python -m app.evaluation.runner --version v1
python -m app.evaluation.runner --version v2
```

Use `--overwrite` only when you intentionally want to replace an existing run's
artifacts. The runner writes JSON metrics, error analysis, and Markdown reports
into `reports/v1/`, `reports/v2/`, and `reports/comparison/` without
consulting development gold during inference.

## Phase 5 backend persistence

The project now includes the Phase 5 FastAPI backend layer with:

- `POST /classify` for a single answer
- `POST /evaluation-runs` for an evaluation execution
- `GET /evaluation-runs/{id}` to fetch saved run details

The API uses the existing claim extractor and classifier and stores run,
answer-result, and claim-result records in PostgreSQL through SQLAlchemy.
The database layer validates `DATABASE_URL` at startup and never prints or
commits secrets.

## Start the API

```powershell
uvicorn app.main:app --reload
```

Then check `http://127.0.0.1:8000/health`; it returns `{"status":"ok"}`.

## Current limitations and later phases

Phase 3 includes the catalogue-grounded extraction/classification workflow,
but evaluation is still a separate live run. The project now includes a
reproducible Phase 4 evaluation harness for frozen Version 1 and Version 2
runs, exact-span scoring, metric calculation, error analysis, and report
writing. Development gold remains separate from inference and is never passed
to the model. Phase 5 persistence and Phase 6 Langfuse integration remain out
of scope.
