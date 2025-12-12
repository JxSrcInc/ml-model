from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): ...
    class Component:
        def run(self, ctx: Context): return ctx


@dataclass
class AccuracyCalculator(Component):
    """
    Multiclass accuracy evaluator.

    Expects in ctx:
        - y:           true labels (N,)
        - pred_label:  predicted labels (N,)

    Writes into ctx:
        - metric_accuracy             (global accuracy)
        - metric_per_class_accuracy   dict[class -> accuracy]
        - metric_per_class_count      dict[class -> count]
        - metric_per_class_correct    dict[class -> correct]
        - metric_precision            dict[class -> precision]
        - metric_recall               dict[class -> recall]
    """

    y_key: str = "y"
    pred_label_key: str = "pred_label"

    def run(self, ctx: Context) -> Context:
        if self.y_key not in ctx:
            raise KeyError(f"AccuracyCalculator requires ctx['{self.y_key}']")
        if self.pred_label_key not in ctx:
            raise KeyError(f"AccuracyCalculator requires ctx['{self.pred_label_key}']")

        y_true = np.asarray(ctx[self.y_key])
        y_pred = np.asarray(ctx[self.pred_label_key])

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Length mismatch: y has {y_true.shape[0]} samples, "
                f"pred has {y_pred.shape[0]}"
            )

        # -------------------------------------------
        # GLOBAL ACCURACY
        # -------------------------------------------
        global_acc = float((y_true == y_pred).mean())
        ctx["metric_accuracy"] = global_acc

        # -------------------------------------------
        # PER-CLASS METRICS
        # -------------------------------------------
        classes = np.unique(y_true)

        per_class_count = {}
        per_class_correct = {}
        per_class_accuracy = {}
        per_class_precision = {}
        per_class_recall = {}

        for c in classes:
            mask = (y_true == c)
            count = mask.sum()
            correct = (y_pred[mask] == c).sum()

            per_class_count[int(c)] = int(count)
            per_class_correct[int(c)] = int(correct)

            # accuracy = correct / count
            per_class_accuracy[int(c)] = float(correct / count) if count > 0 else 0.0

            # precision = TP / (TP + FP)
            tp = correct
            fp = ((y_pred == c) & (y_true != c)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            per_class_precision[int(c)] = float(precision)

            # recall = TP / (TP + FN)
            fn = ((y_pred != c) & (y_true == c)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class_recall[int(c)] = float(recall)

        ctx["metric_per_class_count"] = per_class_count
        ctx["metric_per_class_correct"] = per_class_correct
        ctx["metric_per_class_accuracy"] = per_class_accuracy
        ctx["metric_precision"] = per_class_precision
        ctx["metric_recall"] = per_class_recall

        # -------------------------------------------
        # PRINT SUMMARY
        # -------------------------------------------
        print(f"\n[AccuracyCalculator] Global Accuracy: {global_acc:.4f}")
        print("Per-class metrics:")
        for c in classes:
            c = int(c)
            print(f"  Class {c}: "
                  f"Acc={per_class_accuracy[c]:.4f}, "
                  f"Precision={per_class_precision[c]:.4f}, "
                  f"Recall={per_class_recall[c]:.4f}, "
                  f"Count={per_class_count[c]}, "
                  f"Correct={per_class_correct[c]}")

        return ctx
