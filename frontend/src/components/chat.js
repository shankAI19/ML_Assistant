import { markdownFragment } from "../markdown.js";
import { element } from "../state.js";

export function renderChat(messages, busy) {
  const panel = document.querySelector("#messages");
  const nearBottom =
    panel.scrollHeight - panel.scrollTop - panel.clientHeight < 100;
  panel.replaceChildren();
  if (!messages.length) {
    const greeting = element("div", "welcome");
    greeting.append(
      element("h3", "", "Let’s make sense of your CSV."),
      element(
        "p",
        "",
        "Upload a CSV above and tell me what you want to predict.",
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
      messageBody(message),
    );
    panel.append(item);
  }
  if (busy)
    panel.append(element("div", "thinking", "Working through the next step…"));
  if (nearBottom || messages.length < 3) panel.scrollTop = panel.scrollHeight;
}

function messageBody(message) {
  const body = element("div", "message-body");
  if (message.role === "assistant")
    body.append(markdownFragment(message.content));
  else body.textContent = message.content;
  return body;
}
