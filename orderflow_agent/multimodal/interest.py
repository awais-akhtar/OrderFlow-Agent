"""Optional demo interest model that requires explicit labelled interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

from orderflow_agent.catalog import CatalogItem

from .intelligence import MenuIntelligence


@dataclass(frozen=True)
class InterestModelReport:
    sample_count: int
    folds: int
    mae: float
    dataset_label: str
    encoder: str


class MenuInterestModel:
    def __init__(self, intelligence: MenuIntelligence) -> None:
        self.intelligence = intelligence
        self.model: Ridge | None = None
        self.report: InterestModelReport | None = None

    def fit(
        self,
        labels: Mapping[str, float],
        *,
        dataset_label: str,
    ) -> InterestModelReport:
        if dataset_label.casefold() not in {"synthetic", "demo", "synthetic-demo"}:
            raise ValueError("The dataset label must explicitly identify demo data, or be replaced with a documented real source.")
        items = [item for item in self.intelligence.catalog.active_items if item.sku in labels]
        if len(items) < 6:
            raise ValueError("At least six labelled menu items are required.")
        targets = np.asarray([float(labels[item.sku]) for item in items], dtype=np.float64)
        if np.allclose(targets, targets[0]):
            raise ValueError("Interest labels need variation.")
        features = np.vstack([self.intelligence.representation(item).joint for item in items])
        folds = min(5, len(items))
        predictions = cross_val_predict(Ridge(alpha=1.0), features, targets, cv=KFold(folds, shuffle=True, random_state=42))
        self.model = Ridge(alpha=1.0).fit(features, targets)
        self.report = InterestModelReport(
            sample_count=len(items),
            folds=folds,
            mae=round(float(mean_absolute_error(targets, predictions)), 5),
            dataset_label=dataset_label,
            encoder=self.intelligence.backend.name,
        )
        return self.report

    def predict(self, item: CatalogItem) -> float:
        if self.model is None:
            raise ValueError("Fit the interest model on labelled data before predicting.")
        features = self.intelligence.representation(item).joint.reshape(1, -1)
        return float(self.model.predict(features)[0])
