import { element } from "../state.js";

export function renderApproval(pending, onDecision) {
  const panel = document.querySelector("#approval-panel");
  panel.replaceChildren();
  panel.hidden = !pending;
  if (!pending) return;
  panel.append(element("div", "eyebrow", "REVIEW YOUR TRAINING ROUND"));
  for (const action of pending.actions) {
    const task = action.args.request;
    panel.append(
      element("h3", "", "Training plan"),
      element("p", "", task.explanation),
    );
    if (task.plan) {
      const plan = task.plan;
      panel.append(
        element(
          "p",
          "plan-summary",
          `${plan.model} · target: ${plan.target} · positive: ${plan.positive_label} · ${plan.split} split · threshold: ${plan.threshold} · weighting: ${plan.class_weighted ? "on" : "off"}`,
        ),
      );
      const fields = [
        [
          "Train / validation / test",
          `${Math.round((1 - (plan.validation_fraction ?? 0.2) - (plan.test_fraction ?? 0.2)) * 100)}% / ${Math.round((plan.validation_fraction ?? 0.2) * 100)}% / ${Math.round((plan.test_fraction ?? 0.2) * 100)}%`,
        ],
        ["Excluded columns", plan.excluded_columns?.join(", ") || "None"],
        ["Feature expansion", plan.feature_expansion || "none"],
        ["Polynomial degree", plan.polynomial_degree ?? 2],
        [
          "Focus metrics",
          (
            plan.focus_metrics || [
              "average_precision",
              "precision",
              "recall",
              "f1",
            ]
          ).join(", "),
        ],
        [
          "Preprocessing",
          "Training-only imputation, categorical encoding, numeric scaling for logistic regression",
        ],
      ];
      for (const [label, value] of fields)
        panel.append(element("p", "plan-detail", `${label}: ${value}`));
      const recipe = element("details");
      recipe.append(
        element("summary", "", "Full training recipe"),
        element("pre", "code", JSON.stringify(plan, null, 2)),
      );
      panel.append(recipe);
    }
    const details = element("details");
    details.append(
      element("summary", "", "Review the code"),
      element("pre", "code", task.code),
    );
    panel.append(details);
  }
  const label = element(
    "label",
    "small",
    "Questions or changes? Add feedback before requesting a revision.",
  );
  const feedback = element("textarea");
  feedback.rows = 2;
  feedback.maxLength = 3000;
  feedback.setAttribute("aria-label", "Revision feedback");
  const actions = element("div", "approval-actions");
  const approve = element(
    "button",
    "button primary",
    pending.actions.length > 1 ? "Approve all shown actions" : "Approve & run",
  );
  const revise = element("button", "button secondary", "Request revision");
  for (const [button, decision] of [
    [approve, "approve"],
    [revise, "reject"],
  ]) {
    button.addEventListener("click", async () => {
      approve.disabled = revise.disabled = true;
      try {
        await onDecision({
          approval_id: pending.id,
          decision,
          feedback: feedback.value,
        });
      } finally {
        approve.disabled = revise.disabled = false;
      }
    });
  }
  actions.append(approve, revise);
  panel.append(label, feedback, actions);
}
