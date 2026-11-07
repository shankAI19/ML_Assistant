export function showResults(visible) {
  document.querySelector(".work-area").classList.toggle("has-results", visible);
  document.querySelector(".results-panel").hidden = !visible;
  window.dispatchEvent(new window.Event("resize"));
}
