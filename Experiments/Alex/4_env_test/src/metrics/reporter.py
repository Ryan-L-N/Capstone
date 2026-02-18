"""Summary statistics and visualization generator.

Reads JSONL episode files and produces:
  - summary.csv: aggregate statistics per policy per environment
  - Plots: completion rates, progress boxplots, fall heatmaps, stability curves

Statistical tests:
  - z-test for completion rate differences
  - t-test for continuous metric differences
  - Cohen's d for effect size

Usage:
    python src/metrics/reporter.py --input results/
"""

# TODO: Implementation
# - generate_report(input_dir, output_dir)
# - load_episodes(input_dir) -> DataFrame
# - compute_summary_statistics(df) -> summary table
# - plot_completion_rates(df, output_dir)
# - plot_progress_boxplots(df, output_dir)
# - plot_fall_heatmap(df, output_dir)
# - plot_stability_by_zone(df, output_dir)
# - run_statistical_tests(df) -> p-values, effect sizes


def generate_report(input_dir, output_dir=None):
    """Generate summary CSV and plots from JSONL episode data."""
    raise NotImplementedError("TODO: Implement report generator")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--input", required=True, help="Directory containing JSONL episode files")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()
    generate_report(args.input, args.output or args.input)
