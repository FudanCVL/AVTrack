"""Agent module with factory and registry."""

from typing import Dict, Type

from omegaconf import DictConfig

from .base_agent import BaseAgent
from .sep_agent import SepAgent

try:
    from .face_agent import FaceAgent
except ImportError:
    FaceAgent = None  # deepface not installed

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "base": BaseAgent,
    "sep": SepAgent,
}
if FaceAgent is not None:
    AGENT_REGISTRY["face"] = FaceAgent


def get_agent(cfg: DictConfig) -> BaseAgent:
    """Create agent instance from config.

    Args:
        cfg: Full Hydra config with ``agent.type`` key.

    Returns:
        Instantiated agent.

    Raises:
        ValueError: If agent type is not in the registry.
    """
    agent_type = cfg.agent.type
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[agent_type](cfg)


__all__ = ["BaseAgent", "FaceAgent", "SepAgent", "get_agent", "AGENT_REGISTRY"]
