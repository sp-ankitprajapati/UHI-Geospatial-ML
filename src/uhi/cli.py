"""Command-line entry point; no network calls occur during analysis."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible Hyderabad two-scene analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Analyze downloaded Kaggle v1 inputs")
    run_parser.add_argument("--data-root", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--block-size", type=float, default=1000)
    run_parser.add_argument("--buffer", type=float, default=300)
    run_parser.add_argument("--no-plots", action="store_true")
    fetch_parser = commands.add_parser("fetch", help="Download pinned Kaggle v1 inputs")
    fetch_parser.add_argument("--data-root", required=True)
    fetch_parser.add_argument("--workers", type=int, default=4)
    sample_parser = commands.add_parser("sample", help="Evaluate bundled real labeled samples without raster downloads")
    sample_parser.add_argument("--data-root", default="data/sample")
    sample_parser.add_argument("--output", default="artifacts/sample")
    args = parser.parse_args(argv)
    if args.command == "run":
        from .pipeline import run
        run(args.data_root, args.output, args.block_size, args.buffer, not args.no_plots)
    elif args.command == "fetch":
        from .download import fetch
        fetch(args.data_root, args.workers)
    else:
        from pathlib import Path
        import pandas as pd
        from .evaluation import evaluate
        from .features import FEATURE_NAMES, validate_training
        from .pipeline import write_json
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        results = {}
        for year in (2016, 2022):
            frame = validate_training(pd.read_csv(Path(args.data_root) / f"training_samples_{year}.csv"), FEATURE_NAMES)
            result, predictions, assignments = evaluate(frame[list(FEATURE_NAMES)], frame["class"], frame[["x_m", "y_m"]])
            results[year] = result
            predictions.to_csv(output / f"predictions_{year}.csv", index=False)
            assignments.to_csv(output / f"split_assignments_{year}.csv", index=False)
        write_json(output / "evaluation.json", results)
        print(f"Sample evaluation complete: {output.resolve()}")


if __name__ == "__main__":
    main()
