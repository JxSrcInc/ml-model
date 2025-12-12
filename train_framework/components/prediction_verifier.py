from __future__ import annotations
from dataclasses import dataclass

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): ...
    class Component:
        def run(self, ctx: Context): return ctx


@dataclass
class PredictionVerifier(Component):
    """
    Dispatcher that selects the right evaluator based on ctx['training_model'].

    Expects in ctx:
        - training_model: "classification" or "regression"
        - plus whatever the underlying evaluator needs.

    Uses:
        - classifier_evaluator for "classification"
        - regression_evaluator for "regression"
    """

    classifier_evaluator: Component
    regression_evaluator: Component
    training_model_key: str = "training_model"

    def run(self, ctx: Context) -> Context:
        model_type = str(ctx.get(self.training_model_key, "")).lower()
        if model_type == "classification":
            if self.classifier_evaluator is None:
                raise RuntimeError("PredictionVerifier: no classifier_evaluator set")
            print("[PredictionVerifier] Using classifier evaluator.")
            return self.classifier_evaluator.run(ctx)

        elif model_type == "regression":
            if self.regression_evaluator is None:
                raise RuntimeError("PredictionVerifier: no regression_evaluator set")
            print("[PredictionVerifier] Using regression evaluator.")
            return self.regression_evaluator.run(ctx)

        else:
            raise ValueError(
                f"PredictionVerifier: unknown training_model='{model_type}'. "
                "Expected 'classification' or 'regression'."
            )
