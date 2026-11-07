import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<div id='messages'></div>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { markdownFragment } = await import("../src/markdown.js");
const { renderChat } = await import("../src/components/chat.js");

test("assistant Markdown renders headings, emphasis, tables and code", () => {
  const panel = document.createElement("div");
  panel.append(markdownFragment("## Findings\n\n**Recall** and *precision*\n\n| Metric | Value |\n| --- | --- |\n| Recall | 0.8 |\n\n```python\nprint('hello')\n```"));
  assert.equal(panel.querySelector("h2").textContent, "Findings");
  assert.equal(panel.querySelector("strong").textContent, "Recall");
  assert.equal(panel.querySelector("td").textContent, "Recall");
  assert.match(panel.querySelector("pre code").textContent, /print/);
});

test("Markdown strips scripts, event handlers, remote images and unsafe URLs", () => {
  const panel = document.createElement("div");
  panel.append(markdownFragment('<script>alert(1)</script>\n<img src="https://example.com/track" onerror="alert(1)">\n<a href="javascript:alert(1)" onclick="alert(1)">click</a>\n<iframe src="https://example.com"></iframe>'));
  assert.equal(panel.querySelector("script,img,iframe"), null);
  assert.equal(panel.querySelector("[onclick],[onerror]"), null);
  assert.equal(panel.querySelector('a[href^="javascript:"]'), null);
});

test("user messages stay plain text while assistant messages render Markdown", () => {
  renderChat([{ role: "user", content: "<b>user</b>" }, { role: "assistant", content: "**assistant**" }], false);
  assert.equal(document.querySelector(".user b"), null);
  assert.equal(document.querySelector(".user .message-body").textContent, "<b>user</b>");
  assert.equal(document.querySelector(".assistant strong").textContent, "assistant");
});
