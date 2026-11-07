import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { File } from "node:buffer";
import { JSDOM } from "jsdom";
import { state } from "../src/state.js";

globalThis.__APP_CONFIG__ = {
  appTitle: "ML Assistant",
  apiBaseUrl: "/api",
  maxUploadMb: 25,
};
const { wireUploads, renderUploads } =
  await import("../src/components/upload.js");

async function setup() {
  const html = await readFile(
    new URL("../index.html", import.meta.url),
    "utf8",
  );
  const dom = new JSDOM(html);
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  state.session = {
    id: "session",
    experiment_id: "one",
    busy: false,
    pending: null,
    dataset_uploaded: false,
    prediction_uploaded: false,
    finalized: false,
  };
  state.stopped = state.uploading = false;
  const calls = [],
    errors = [];
  let successful = true;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, file: options.body.get("file") });
    return {
      ok: successful,
      status: successful ? 200 : 400,
      json: async () => ({ detail: "The CSV is empty" }),
    };
  };
  const dialog = document.querySelector("#upload-dialog");
  dialog.showModal = () => dialog.setAttribute("open", "");
  dialog.close = (value) => {
    dialog.returnValue = value;
    dialog.removeAttribute("open");
    dialog.dispatchEvent(new window.Event("close"));
  };
  wireUploads(
    async () => {
      const { url, file } = calls.at(-1);
      if (url.includes("kind=training")) {
        state.session.dataset_uploaded = true;
        state.session.training_filename = file.name;
      } else {
        state.session.prediction_uploaded = true;
        state.session.prediction_filename = file.name;
      }
    },
    (error) => errors.push(error),
  );
  return {
    dialog,
    calls,
    errors,
    fail: () => {
      successful = false;
    },
  };
}

async function chooseFile(dialog, purpose, name = "data.csv") {
  const input = document.querySelector("#csv-file");
  Object.defineProperty(input, "files", {
    configurable: true,
    value: [new File(["x,y\n1,0\n"], name)],
  });
  input.dispatchEvent(new window.Event("change"));
  assert.equal(dialog.open, true);
  dialog.close(purpose);
  await new Promise((resolve) => setImmediate(resolve));
}

test("one CSV button supports initial training, training replacement and early predictions", async () => {
  const { dialog, calls, errors } = await setup();
  assert.equal(document.querySelectorAll('input[type="file"]').length, 1);
  await chooseFile(dialog, "training", "train.csv");
  assert.equal(calls[0].url, "/api/uploads?kind=training");
  assert.equal(document.querySelector("#csv-file").disabled, false);
  await chooseFile(dialog, "training", "new-training.csv");
  assert.equal(
    calls[1].url,
    "/api/uploads?kind=training&replace_training=true",
  );
  assert.equal(
    document.querySelector("#training-replacement-note").hidden,
    false,
  );
  assert.match(
    document.querySelector("#training-status").textContent,
    /new-training.csv/,
  );
  await chooseFile(dialog, "prediction", "predict.csv");
  assert.equal(calls[2].url, "/api/uploads?kind=prediction");
  assert.equal(document.querySelector("#prediction-status").hidden, false);
  assert.match(
    document.querySelector("#prediction-status").textContent,
    /queued until finalization/,
  );
  assert.equal(document.querySelector("#csv-file").disabled, false);
  assert.deepEqual(errors, []);
});

test("cancel and upload failure leave the shared button available", async () => {
  const { dialog, calls, errors, fail } = await setup();
  await chooseFile(dialog, "cancel");
  assert.equal(calls.length, 0);
  assert.equal(document.querySelector("#csv-file").disabled, false);
  fail();
  await chooseFile(dialog, "prediction");
  assert.equal(errors.length, 1);
  assert.equal(document.querySelector("#csv-file").disabled, false);
  assert.equal(state.session.prediction_uploaded, false);
});

test("pending approval does not lock uploads; only an active task temporarily does", async () => {
  await setup();
  state.session.pending = { id: "approval" };
  state.session.dataset_uploaded = true;
  renderUploads();
  assert.equal(document.querySelector("#csv-file").disabled, false);
  state.session.busy = true;
  renderUploads();
  assert.equal(document.querySelector("#csv-file").disabled, true);
  assert.match(
    document.querySelector("#upload-hint").textContent,
    /Task running/,
  );
  state.session.busy = false;
  renderUploads();
  assert.equal(document.querySelector("#csv-file").disabled, false);
});
