"""Print locally saved responses for a W&B study; run on the benchmark machine."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default=
                        "maorblumberg-tel-aviv-university/inference-optimization/wchsr4i5")
    parser.add_argument("--results-dir", type=Path, default=Path.cwd(),
                        help="Directory to search recursively for measured JSONL files")
    parser.add_argument("--file", type=Path, help="Inspect an exact file without W&B access")
    parser.add_argument("--limit", type=int, default=10, help="Responses per saved trial; 0 prints all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.file:
        files = [args.file]
    else:
        try:
            import wandb
            run = wandb.Api().run(args.run)
        except Exception as exc:
            parser.exit(1, f"Cannot read W&B run: {exc}\nUse --file to inspect a local measured JSONL directly.\n")
        study = run.config.get("study_name")
        if not study:
            parser.exit(1, "Run has no study_name. Use --file to select the result file.\n")
        print(f"W&B run: {args.run}\nStudy: {study}")
        # The framework suffix follows the study name in measured file paths.
        # Search relative paths as study names can contain model-name slashes.
        files = sorted(p for p in args.results_dir.rglob("*-measured.jsonl")
                       if p.relative_to(args.results_dir).as_posix().endswith(
                           f"{study}-{run.config.get('phase', '')}-{run.config.get('framework', '')}"
                           f"-c{run.config.get('concurrency', '')}-measured.jsonl"))
        if not files:
            # Some campaign versions use a shorter run label.
            files = sorted(p for p in args.results_dir.rglob("*-measured.jsonl")
                           if study in p.relative_to(args.results_dir).as_posix())
    if not files:
        parser.exit(1, "No matching local results. Run this on the Jupyter machine/container, "
                    "or copy its results and specify --results-dir. Generated text is saved locally.\n")
    count = 0
    for path in files:
        print(f"\nFILE: {path}")
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid/incomplete line {line_number}", file=sys.stderr)
                    continue
                details = row.get("results", {})
                texts = details.get("generated_texts") or []
                errors = details.get("errors") or []
                trial = row.get("optimization", {}).get("trial_number", "unknown")
                print(f"\nTrial {trial} | line {line_number} | saved responses: {len(texts)}")
                for index, text in enumerate(texts):
                    if args.limit and index >= args.limit:
                        break
                    print(f"\n--- Request {index + 1} ---\n{text}")
                    if index < len(errors) and errors[index]:
                        print(f"Error: {errors[index]}")
                    count += 1
    if not count:
        print("No saved responses. Future runs need SAVE_REQUEST_OUTPUTS=True.")


if __name__ == "__main__":
    main()
