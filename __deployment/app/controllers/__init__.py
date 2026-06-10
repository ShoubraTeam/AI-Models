__all__ = [
    "FeatureController",
    "AgentController",
    "WeaviateController",
]


def __getattr__(name: str):
    if name == "FeatureController":
        from .feature_controller import FeatureController

        return FeatureController

    if name == "AgentController":
        from .agents_controller import AgentController

        return AgentController

    if name == "WeaviateController":
        from .weaviate_controller import WeaviateController

        return WeaviateController

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
