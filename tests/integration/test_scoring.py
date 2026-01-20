import asyncio
import json
import subprocess
import sys
from pathlib import Path

# ruff: noqa: E402
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import generate


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    print("cmd:", " ".join(map(str, cmd)))
    print("returncode:", p.returncode)
    print("stdout:\n", p.stdout)
    print("stderr:\n", p.stderr)
    p.check_returncode()
    return p


def find_latest_subdir(parent: Path) -> Path:
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {parent}")
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def generate_one_persona(
    persona_name: str,
    member_model: str,
    provider_model: str,
    conversations_root: Path,
    turns: int,
    runs: int,
    temp_member: float,
    temp_provider: float,
) -> Path:
    conversations_root.mkdir(parents=True, exist_ok=True)

    persona_model_config = {
        "model": member_model,
        "temperature": temp_member,
    }
    agent_model_config = {
        "model": provider_model,
        "name": provider_model,
        "temperature": temp_provider,
    }

    asyncio.run(
        generate.main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            persona_names=[persona_name],  # only one persona
            max_turns=turns,
            runs_per_prompt=runs,
            folder_name=str(conversations_root),
            max_concurrent=1,
            verbose=True,
        )
    )
    return find_latest_subdir(conversations_root)


def judge_conversations(
    conversations_dir: Path,
    judge_model: str,
    repo_root: Path,
) -> Path:
    run_cmd(
        [sys.executable, "judge.py", "-f", str(conversations_dir), "-j", judge_model],
        cwd=repo_root,
    )

    eval_root = repo_root / "evaluations"
    return find_latest_subdir(eval_root)


def score_evaluation(
    eval_dir: Path,
    repo_root: Path,
) -> dict:
    results_csv = eval_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv in {eval_dir}")

    p = run_cmd(
        [sys.executable, "-m", "judge.score", "-r", str(results_csv)],
        cwd=repo_root,
    )

    out = p.stdout.strip()
    start, end = out.find("{"), out.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"No JSON found in judge.score output:\n{out}")

    return json.loads(out[start : end + 1])


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent.parents[0]

    CONVERSATIONS_ROOT = REPO_ROOT / "conversations"

    PERSONAS = [
        # "Alix",
        # "Noah",
        "Sky",
    ]

    MEMBER_MODEL = "gpt-4o"
    PROVIDER_MODEL = "claude-opus-4-1-20250805"
    JUDGE_MODEL = "gpt-4o"

    TURNS = 10
    RUNS_PER_PERSONA = 5
    TEMP_MEMBER = 1.0
    TEMP_PROVIDER = 1.0

    all_scores = {}

    for persona in PERSONAS:
        print("\n" + "=" * 80)
        print(f"Running pipeline for persona: {persona}")

        conv_dir = generate_one_persona(
            persona_name=persona,
            member_model=MEMBER_MODEL,
            provider_model=PROVIDER_MODEL,
            conversations_root=CONVERSATIONS_ROOT,
            turns=TURNS,
            runs=RUNS_PER_PERSONA,
            temp_member=TEMP_MEMBER,
            temp_provider=TEMP_PROVIDER,
        )

        eval_dir = judge_conversations(
            conversations_dir=conv_dir,
            judge_model=JUDGE_MODEL,
            repo_root=REPO_ROOT,
        )

        score_json = score_evaluation(
            eval_dir=eval_dir,
            repo_root=REPO_ROOT,
        )

        all_scores[persona] = score_json

        print(
            f"{persona} → "
            f"best_practice={score_json['aggregates']['overall_best_practice_pct']} | "
            f"damaging={score_json['aggregates']['overall_damaging_pct']} | "
            f"worst_band={score_json['aggregates']['worst_band']}"
        )

    print("\n" + "=" * 80)
    print("DONE. Personas scored:", list(all_scores.keys()))
