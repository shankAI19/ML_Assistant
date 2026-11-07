// Deliberately no localStorage, sessionStorage, or IndexedDB.
export const state = {
  session: null,
  poll: null,
  stopped: false,
  uploading: false,
  signature: "",
  resultsSignature: "",
};

export function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}
