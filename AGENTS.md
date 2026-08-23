# Instructions for Coding Agents

## 1. Interaction with the User

- **Respect Questions:** When asked questions, answer the question and make helpful suggestions. Never start coding without explicit confirmation when asked questions.
- **Provide Summaries:** When performing coding tasks, always provide a brief summary and explanation of your changes. Highlight key findings and decisions you have made autonomously on the way.
- **Ask for Help:** When you keep failing on a task, stop and explain the issue. Ask for human expert opinion and collaboratively solve particularly hard tasks.
- **Confirm External Runs:** Never launch paid training, create or scale a deployment, upload a dataset, or otherwise incur provider cost without explicit user confirmation. Read-only capability checks and local dry runs are allowed.

## 2. General Coding Instructions

- **Planning:** Make a plan and weigh options before starting larger coding tasks.
- **Code Organization:** Prefer loose coupling, high composability, and clear separation of concerns. Keep provider adapters thin so data preparation, ontology validation, and evaluation remain usable offline.
- **Functional Style:** Prefer short, chainable, and pure functions with well-scoped responsibilities.
- **Naming:** Prefer self-explaining and concise names. Add code documentation to inherently complex functions and classes.
- **Python Tooling:** Use `uv` for Python version selection, dependency management, environment synchronization, command execution, locking, and package builds. Keep `pyproject.toml` and `uv.lock` synchronized; do not add parallel requirements or lock files.
- **Tests:** Generate and execute unit tests for production code. Follow the repository's configured pytest and coverage conventions once the Python toolchain exists, and keep provider calls behind fakes or recorded fixtures in unit tests. Verify high coverage with the repository's complete coverage command.
- **Temporary Files:** Never write scratch files to the repository root. Captured command output, logs, intermediate data, provider-ready JSONL, raw predictions, reports, and other generated artifacts belong in the applicable gitignored directory (`temp/`, `data/`, `artifacts/`, `runs/`, or `reports/`). Clean up scratch files that are no longer needed.
- **Secrets:** Keep Fireworks, Hugging Face, Weights & Biases, and other credentials in environment variables or an approved secret manager. Never commit credentials or include them in manifests, reports, fixtures, prompts, or logs.

## 3. Project Documentation

Before executing any task, make yourself familiar with the project:

- **ALWAYS read `README.md`** for the repository purpose, ownership boundaries, and current status.
- **Load the relevant document under `docs/`** before authoring or reviewing the corresponding subsystem. Treat the project documents as the source of truth for accepted architecture and experiment policy.

Update the relevant documentation after larger changes. Architecture, workflows, and experiment decisions belong in `docs/`; `README.md` should remain a concise entry point rather than duplicating them.

## 4. Classifier-Specific Guardrails

- **Atomic Input:** Preserve the invariant that one independent pedagogical task equals one classification unit. Keep all renderings of the same underlying task in the same data split.
- **Repository Boundaries:** `edugraph-dataset` owns released labeled images, official splits, and the public dataset contract; `edugraph-ontology` owns identifiers, dimensions, definitions, versions, and relations; this repository owns classifier prompts, conversion, orchestration, inference, evaluation, and experiment records. Do not duplicate upstream ownership here.
- **Provider and Execution Boundaries:** Keep provider SDKs behind thin adapters and keep provider identity separate from execution mode. Core data, ontology, inference, and evaluation code must not assume Fireworks, serverless execution, or self-hosted execution. Treat provider-hosted dedicated deployments as distinct from both serverless and self-hosted execution when that distinction matters.
- **Version Pinning:** Every run must pin and record the dataset release, ontology version, prompt/schema identity, code commit, provider model/job configuration, and evaluation settings.
- **Ontology Semantics:** Keep explicit model predictions separate from ontology-derived closure. Do not treat `integrates` as logical entailment. Apply only deterministic, documented validation and canonicalization rules.
- **Provider Eligibility:** Fireworks model and training-shape support is volatile. Check the live catalog and require `Tunable: true` plus a compatible vision training surface before preparing or launching a job.
- **Determinism:** Make conversion, canonical serialization, metrics, and report inputs deterministic. Preserve raw predictions alongside canonicalized predictions so post-processing gains remain measurable.
- **Evaluation First:** Use exact-set match as the primary outcome and report precision, recall, F1, per-dimension, per-view, frequency-slice, invalid-output, latency, and cost diagnostics. Do not promote a model from validation loss alone.
