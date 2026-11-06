import { element } from "../state.js";

export function renderChat(messages, busy) {
  const panel = document.querySelector("#messages");
  const nearBottom =
    panel.scrollHeight - panel.scrollTop - panel.clientHeight < 100;
  panel.replaceChildren();
  if (!messages.length) {
    const greeting = element("div", "welcome");
    greeting.append(
      element("span", "welcome-symbol", "✳"),
      element("h3", "", "Let’s make sense of your CSV."),
      element(
        "p",
        "",
        "Tell me your goal—or upload your data and we’ll find a starting point together. I’ll explain each step before asking you to approve it.",
      ),
    );
    panel.append(greeting);
  }
  for (const message of messages) {
    const item = element("article", `message ${message.role}`);
    item.append(
      element(
        "div",
        "message-label",
        message.role === "assistant"
          ? "ML GUIDE"
          : message.role === "user"
            ? "YOU"
            : "WORKSPACE",
      ),
      element("div", "message-body", message.content),
    );
    panel.append(item);
  }
  if (busy)
    panel.append(element("div", "thinking", "Working through the next step…"));
  if (nearBottom || messages.length < 3) panel.scrollTop = panel.scrollHeight;
}
