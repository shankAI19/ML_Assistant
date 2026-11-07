import { marked } from "marked";
import DOMPurify from "dompurify";

export function markdownFragment(source) {
  return DOMPurify.sanitize(marked.parse(String(source), { breaks: true }), {
    RETURN_DOM_FRAGMENT: true,
    USE_PROFILES: { html: true },
    FORBID_TAGS: [
      "img",
      "iframe",
      "form",
      "input",
      "style",
      "svg",
      "video",
      "audio",
    ],
    FORBID_ATTR: ["style", "id", "name"],
  });
}
