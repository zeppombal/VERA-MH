# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where versioning is applied.

## [1.1.0] - 2026-04-22

### Breaking / migration

- **`data/personas.tsv` schema** — Replaced prior columns (e.g. race/ethnicity, pronouns, combined mental-health context, sample prompts) with structured fields including gender bands, suicide-risk descriptors (full + short), remote history, diagnoses/symptoms, treatment access, stigma, financial stress, isolation, discrimination exposure, triggers/stressors, and a **Final Seed Phrase** (clinician-style anchor; not meant to be echoed verbatim by the user-agent). Custom persona sheets must be migrated to match `data/persona_prompt_template.txt` placeholders.

### Data and prompts

- Expanded built-in personas from **10 -> 100** with broader topic and risk coverage.
- **`data/persona_prompt_template.txt`** — Reworked backstory block, “seed phrase” guidance (replaces “sample prompts” behavior), provider-first wording, anti-medical-jargon instructions, optional partial disclosure of triggers, selective response to multi-part provider messages, em-dash avoidance.
- **`data/persona_prompt_reminder.txt`** (new) — Per-turn reminder appended before the latest provider message to reduce user-agent role drift and clinical register leakage.

### Rubric and scoring

- **`data/rubric.tsv`** — Substantive revision from stakeholder feedback:
  - **Guides to Human Care** — Less blanket penalization; more context (e.g. user already engaged with crisis resources; tie absence of distress-tolerance strategies more tightly to **current** suicidal urgency in-thread).
  - **High Potential for Harm vs Suboptimal** — Clearer boundaries (e.g. barriers to using crisis resources shifted toward suboptimal; **no** crisis contact information remains high harm).
  - **Inter-dimension dependence** — Reduced coupling (e.g. high harm on **Detects Potential Risk** no longer automatically forces high harm on **Confirms Risk**).
- **Interpretation** — Aggregate scores are not comparable to pre-1.1 without versioning; observed **small upward shifts** (~1–7 points) on general LLM aggregates vs the prior rubric in internal checks.

### Runtime, CLI, and pipeline

- **LLM calls** — Retry + timeout behavior (default **max 3 retries** with delay between attempts; configurable where exposed by CLI/config).
- **Fault tolerance** — **skip** conversations or judge jobs that error instead of returning the error as LLM's response
- **Default output layout** — `README.md` documents timestamped **`p_*__a_*__t*__r*__*`** folders (by default under **`output/`**), with transcripts in **`conversations/`** inside that folder; batch judging writes **`j_*__*`** under **`evaluations/`** next to the generation run when using the nested layout (see `README` / `judge.py` `--help` for `-f` / `-o` defaults, which evolved across revisions).
- **`generate.py`** — **`--resume`**: point the generation path argument at an existing **`p_*`** run directory so persona/run pairs that already have transcripts are skipped; validates matching user/provider models, turns, and runs. **Flag name** for that path is **`--output`** in the README-first CLI and **`--folder-name` / `-f`** in older revisions—check **`--help`** on your tree.
- **`judge.py`** — **`--resume`**: pass **`-o`** as the **existing** **`j_*__*`** evaluation directory (not only the parent `evaluations/`). Skips `(conversation, judge, instance)` jobs with an existing `.tsv`; rebuilds `results.csv` from TSVs. Not supported with `-c` / `--conversation`. **`-f`** should point at the generation run folder (or another folder that resolves conversation paths the same way—see `README`).
- **`run_pipeline.py`** — **`--resume-generate`** / **`--resume-judge`**: **`--output`** means different things depending on mode; using **both** resume flags requires **exactly one** **`j_*`** under **`p_*/evaluations/`** so the pipeline can pick the evaluation run to continue (see `README`).

### Automation (recommended scoring; when scripts are present)

- **`scripts/run_recommended_vera_pipeline.sh`** — Runs the recommended multi-user-agents, 30-turn, multi-judge profile described under **Recommended settings** in `README.md`, then pools scores (environment variables documented in-script).
- **`scripts/pool_vera_scores.py`** — Merge multiple existing **`j_*`** evaluation directories into a **`j_pooled__…`** folder with merged **`results.csv`**, metadata, and score artifacts (see **`python3 scripts/pool_vera_scores.py --help`**).

### Outputs, logging, and repo hygiene

- **Judge logs** — One log file per **conversation × judge model × instance** (parallel stems to per-conversation **`.tsv`** files). Default root is **`judge_logs/`** in the working directory (override with **`VERA_JUDGE_LOGS_ROOT`**); nested-run docs in **`README.md`** may additionally describe a **`logs/`** tree beside **`results.csv`** depending on revision—prefer env + `--help` for your checkout.
- **Run directory layout** — Co-locates generation, evaluations, scoring inputs/outputs for a single **`p_*`** run where the nested layout is used (see `README` / pipeline summary).

### Documentation

- **`README.md`** — Expanded getting started: **Recommended settings** for comparable scores; **Reliable VERA score (automated)** (`run_recommended_vera_pipeline.sh`, pooling); **`output/`**-centric paths; **`run_pipeline.py`** resume semantics (**`--output`** overload, single-**`j_*`** constraint when resuming both stages); **Additional Resources** (e.g. arXiv papers); **Connecting your own LLM, Agent, or API**; `generate.py` / `judge.py` **`--resume`**; contributor persona TSV field list; testing/`live` marker notes as applicable.

[1.1.0]: https://github.com/SpringCare/VERA-MH/releases/tag/v1.1.0
