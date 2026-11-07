import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskKind = Literal["profile", "train", "evaluate", "predict", "report"]


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(min_length=1, max_length=200)
    positive_label: str = Field(min_length=1, max_length=200)
    model: Literal["logistic", "xgboost", "lightgbm", "random_forest"] = "logistic"
    excluded_columns: list[str] = Field(default_factory=list, max_length=100)
    class_weighted: bool = True
    threshold: float = Field(0.5, gt=0, lt=1)
    split: Literal["stratified", "chronological", "group"] = "stratified"
    split_column: str | None = None
    seed: int = Field(42, ge=0, le=2**31 - 1)
    validation_fraction: float = Field(0.2, gt=0, lt=1)
    test_fraction: float = Field(0.2, ge=0, lt=1)
    feature_expansion: Literal["none", "polynomial", "interaction_only"] = "none"
    polynomial_degree: int = Field(2, ge=2, le=3)
    focus_metrics: list[
        Literal[
            "average_precision",
            "precision",
            "recall",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "roc_auc",
            "alert_rate",
        ]
    ] = Field(
        default_factory=lambda: ["average_precision", "precision", "recall", "f1"],
        min_length=1,
        max_length=9,
    )
    # Keep the demo's search space small and explicit.
    trees: int = Field(150, ge=20, le=500)
    max_depth: int = Field(5, ge=2, le=12)
    learning_rate: float = Field(0.05, ge=0.01, le=0.3)
    regularization: float = Field(1.0, ge=0.01, le=100)

    @model_validator(mode="after")
    def check_columns(self):
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("Validation and test fractions must leave training data")
        if self.feature_expansion != "none" and self.model != "logistic":
            raise ValueError("Feature expansion is supported for logistic regression")
        if self.target in self.excluded_columns:
            raise ValueError("The target cannot be an excluded feature")
        if self.split != "stratified" and not self.split_column:
            raise ValueError("Group/time splits need a split_column")
        return self


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: TaskKind
    explanation: str = Field(min_length=10, max_length=3000)
    code: str = Field(min_length=1, max_length=20000)
    plan: ExperimentPlan | None = None
    narrative: str = Field(default="", max_length=20000)

    def payload(self) -> dict:
        return {
            "task": self.task,
            "plan": self.plan.model_dump() if self.plan else None,
            "narrative": self.narrative,
        }

    @model_validator(mode="after")
    def validate_generated_code(self):
        if self.task == "train" and self.plan is None:
            raise ValueError("Training requires a plan")
        try:
            tree = ast.parse(self.code)
            first, second = tree.body
            valid_import = (
                isinstance(first, ast.ImportFrom)
                and first.module == "ml_runtime"
                and first.level == 0
                and len(first.names) == 1
                and first.names[0].name == "execute"
                and first.names[0].asname is None
            )
            call = second.value if isinstance(second, ast.Expr) else None
            valid_call = (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "execute"
                and len(call.args) == 1
                and not call.keywords
            )
            if (
                not valid_import
                or not valid_call
                or not self.matches_payload(ast.literal_eval(call.args[0]))
            ):
                raise ValueError("Code must call execute with exactly the approved task payload")
        except (SyntaxError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "Use only: from ml_runtime import execute; execute(<literal payload>)"
            ) from exc
        return self

    def matches_payload(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"task", "plan", "narrative"}:
            return False
        if payload["plan"] is not None:
            payload = {
                **payload,
                "plan": ExperimentPlan.model_validate(payload["plan"]).model_dump(),
            }
        return payload == self.payload()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)


class ApprovalRequest(BaseModel):
    approval_id: str
    decision: Literal["approve", "reject"]
    feedback: str = Field(default="", max_length=3000)
