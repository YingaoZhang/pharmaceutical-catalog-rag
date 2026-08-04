const DEFAULT_API_BASE = "http://127.0.0.1:8001";

export function getSavedApiBase() {
  const savedApiBase = localStorage.getItem("pharma-rag-api-base");
  return savedApiBase || DEFAULT_API_BASE;
}

export function saveApiBase(value) {
  localStorage.setItem("pharma-rag-api-base", value.trim().replace(/\/$/, ""));
}

function joinUrl(apiBase, path) {
  const base = (apiBase || "").trim().replace(/\/$/, "");
  return `${base}${path}`;
}

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
  }
  return payload;
}

export async function fetchStats(apiBase) {
  const response = await fetch(joinUrl(apiBase, "/api/v1/stats"));
  return parseResponse(response);
}

export async function askQuestion(apiBase, question) {
  const response = await fetch(joinUrl(apiBase, "/api/v1/qa/ask"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      context: {},
    }),
  });
  return parseResponse(response);
}

export async function uploadDocument(apiBase, file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(joinUrl(apiBase, "/api/v1/documents/upload"), {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}
