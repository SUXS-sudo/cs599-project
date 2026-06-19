from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.data_pipeline import build_eval_seed, run_recipe_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned SmartRecipe data artifacts.")
    parser.add_argument("--source", default=str(ROOT / "data" / "recipes.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "recipes_clean.json"))
    parser.add_argument("--eval-output", default=str(ROOT / "data" / "evals" / "pipeline_recipe_seed.jsonl"))
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    report = run_recipe_pipeline(source_path, output_path)

    recipes = json.loads(output_path.read_text(encoding="utf-8"))
    eval_rows = build_eval_seed(recipes)
    eval_output = Path(args.eval_output)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in eval_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report.to_dict(), "eval_output": str(eval_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
