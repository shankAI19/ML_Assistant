import { element } from "../state.js";

export function renderApproval(pending, onDecision) {
  const panel = document.querySelector("#approval-panel");
  panel.replaceChildren();
  panel.hidden = !pending;
  if (!pending) return;
  panel.append(element("div", "eyebrow", "YOUR APPROVAL IS NEEDED"));
  for (const action of pending.actions) {
    const task = action.args.request;
    panel.append(
      element(
        "h3",
        "",
        `${task.task[0].toUpperCase()}${task.task.slice(1)} dataset task`,
      ),
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
