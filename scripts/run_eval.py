import logging

import hydra
from omegaconf import DictConfig

from src.evaluation.eval_runner import evaluate

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """AVTrack evaluation entry point.

    Usage:
        python scripts/run_eval.py output_dir=outputs/base
        python scripts/run_eval.py output_dir=outputs/sep agent=sep
    """
    output_dir = cfg.get("output_dir", f"outputs/{cfg.agent.name}")
    merge_dir = output_dir + "_merged"

    logger.info("Evaluating predictions from: %s", output_dir)
    results = evaluate(cfg, pred_mask_dir=output_dir, pred_mask_merge_dir=merge_dir)

    # Print results in a nice format
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print("========================\n")


if __name__ == "__main__":
    main()
