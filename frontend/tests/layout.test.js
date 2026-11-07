import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
import { showResults } from "../src/layout.js";

test("results appear beside the existing chat and preserve an unsent message", async () => {
  const html = await readFile(
    new URL("../index.html", import.meta.url),
    "utf8",
  );
  const dom = new JSDOM(html);
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  const chat = document.querySelector(".conversation-panel");
  const results = document.querySelector(".results-panel");
  const messages = document.querySelector("#messages");
  messages.textContent = "Explain my validation results";
  document.querySelector("#message").value = "What about recall?";
  assert.equal(results.hidden, true);
  showResults(true);
  assert.equal(results.hidden, false);
  assert.equal(chat.hidden, false);
  assert.equal(
    document.querySelector(".work-area").classList.contains("has-results"),
    true,
  );
  assert.equal(document.querySelector("#message").value, "What about recall?");
  assert.equal(messages.textContent, "Explain my validation results");
  showResults(false);
  assert.equal(results.hidden, true);
  assert.equal(chat.hidden, false);
  assert.equal(
    document.querySelector(".work-area").classList.contains("has-results"),
    false,
  );
  assert.equal(
    document.querySelector("#training-file").closest(".upload-bar") !== null,
    true,
  );
});
