import json
import os
from datetime import datetime


def load_run_info(file_path):
    """Load run information from JSON file"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def compare_metrics(raw_metrics, optimized_metrics):
    """Compare metrics between raw and optimized models"""

    if not raw_metrics or not optimized_metrics:
        print("Error: Missing metrics for comparison")
        return None

    raw_rmse = raw_metrics.get("rmse", float("inf"))
    raw_r2 = raw_metrics.get("r2_score", -float("inf"))

    optimized_rmse = optimized_metrics.get("rmse", float("inf"))
    optimized_r2 = optimized_metrics.get("r2_score", -float("inf"))

    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)

    print(f"\nRaw Model:")
    print(f"  RMSE: {raw_rmse:.4f}")
    print(f"  R² Score: {raw_r2:.4f}")

    print(f"\nOptimized Model:")
    print(f"  RMSE: {optimized_rmse:.4f}")
    print(f"  R² Score: {optimized_r2:.4f}")

    # Calculate improvements
    rmse_improvement = (raw_rmse - optimized_rmse) / raw_rmse * 100
    r2_improvement = (optimized_r2 - raw_r2) / abs(raw_r2) * 100 if raw_r2 != 0 else 0

    print(f"\nImprovement Analysis:")
    print(f"  RMSE improvement: {rmse_improvement:.2f}%")
    print(f"  R² improvement: {r2_improvement:.2f}%")

    # Decision logic
    # Prioritize RMSE (lower is better), but also consider R²
    rmse_threshold = 2.0  # Minimum 2% improvement needed
    r2_threshold = 1.0  # Minimum 1% improvement needed

    if optimized_rmse < raw_rmse and rmse_improvement >= rmse_threshold:
        if optimized_r2 >= raw_r2 or r2_improvement >= r2_threshold:
            winner = "optimized"
            reason = f"Lower RMSE ({rmse_improvement:.2f}% improvement) with comparable/better R²"
        else:
            # RMSE is better but R² is worse - need to decide
            if rmse_improvement > 5.0:  # Significant RMSE improvement
                winner = "optimized"
                reason = f"Significant RMSE improvement ({
                    rmse_improvement:.2f}%) outweighs R² decrease"
            else:
                winner = "Raw"
                reason = f"RMSE improvement ({
                    rmse_improvement:.2f}%) not significant enough to justify R² decrease"
    elif optimized_r2 > raw_r2 and r2_improvement >= r2_threshold:
        if optimized_rmse <= raw_rmse or abs(rmse_improvement) <= 1.0:
            winner = "optimized"
            reason = f"Better R² ({
                r2_improvement:.2f}% improvement) with comparable/better RMSE"
        else:
            winner = "raw"
            reason = f"R² improvement doesn't justify RMSE increase of {
                abs(rmse_improvement):.2f}%"
    else:
        winner = "raw"
        reason = "Optimization did not provide sufficient improvement"

    print(f"\nWINNER: {winner.upper()} MODEL")
    print(f"REASON: {reason}")
    print("=" * 50)

    return {
        "winner": winner,
        "reason": reason,
        "improvements": {
            "rmse_improvement_pct": rmse_improvement,
            "r2_improvement_pct": r2_improvement,
        },
        "raw_metrics": {"rmse": raw_rmse, "r2_score": raw_r2},
        "optimized_metrics": {"rmse": optimized_rmse, "r2_score": optimized_r2},
    }


def main():
    """Main comparison function"""

    # Load run information from both models
    raw_info = load_run_info("raw_run_info.json")
    optimized_info = load_run_info("optimized_run_info.json")

    if not raw_info:
        print("Could not load Raw model information")
        return

    if not optimized_info:
        print("Could not load optimized model information, defaulting to Raw model")
        winner_info = raw_info
        comparison_result = {
            "winner": "Raw",
            "reason": "Optimized model not available",
            "improvements": {"rmse_improvement_pct": 0, "r2_improvement_pct": 0},
            "raw_metrics": raw_info.get("validation_metrics", {}),
            "optimized_metrics": {},
        }
    else:
        # Compare the models
        raw_metrics = raw_info.get("validation_metrics", {})
        optimized_metrics = optimized_info.get("validation_metrics", {})

        comparison_result = compare_metrics(raw_metrics, optimized_metrics)

        if comparison_result is None:
            print("Comparison failed, defaulting to Raw model")
            winner_info = raw_info
            comparison_result = {
                "winner": "Raw",
                "reason": "Comparison failed - using fallback",
                "improvements": {"rmse_improvement_pct": 0, "r2_improvement_pct": 0},
                "raw_metrics": raw_metrics,
                "optimized_metrics": optimized_metrics,
            }
        else:
            # Select winner information
            if comparison_result["winner"] == "optimized":
                winner_info = optimized_info
            else:
                winner_info = raw_info

    # Create the final run info for the winning model
    final_run_info = {
        "mlflow_run_id": winner_info["mlflow_run_id"],
        "mlflow_tracking_uri": winner_info["mlflow_tracking_uri"],
        "validation_metrics": winner_info["validation_metrics"],
        "model_type": winner_info["model_type"],
        "model_uri": winner_info["model_uri"],
        "selection_info": {
            "comparison_date": datetime.now().isoformat(),
            "winner": comparison_result["winner"],
            "selection_reason": comparison_result["reason"],
            "compared_models": ["raw", "optimized"] if optimized_info else ["raw"],
            "improvements": comparison_result["improvements"],
        },
        "all_metrics": {
            "raw": comparison_result["raw_metrics"],
            "optimized": comparison_result["optimized_metrics"],
        },
    }

    # Save the final run info
    os.makedirs("model_artifacts", exist_ok=True)
    with open("model_artifacts/final_run_info.json", "w") as f:
        json.dump(final_run_info, f, indent=2)

    # Also save the detailed comparison
    with open("model_artifacts/comparison_report.json", "w") as f:
        json.dump(comparison_result, f, indent=2)

    print(f"\nComparison completed!")
    print(f"Final run info saved to: model_artifacts/final_run_info.json")
    print(f"Detailed comparison saved to: model_artifacts/comparison_report.json")
    print(f"Selected model: {comparison_result['winner'].upper()}")


if __name__ == "__main__":
    main()
