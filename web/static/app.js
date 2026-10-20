const uploadedDocs = [];

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadButton = uploadForm.querySelector("button[type='submit']");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
let docList = document.getElementById("docList");
const selectionSummary = document.getElementById("selectionSummary");
const answerText = document.getElementById("answerText");
const modeTag = document.getElementById("modeTag");
const answerTypeTag = document.getElementById("answerTypeTag");
let evidenceBody = document.getElementById("evidenceBody");
const debugText = document.getElementById("debugText");
const serviceStatus = document.getElementById("serviceStatus");
const appState = globalThis.RLMQAState || {
  selectedSourceIds(docs) {
    return docs
      .filter((doc) => doc.selected && this.isDocumentAskable(doc))
      .map((doc) => doc.source_id);
  },
  upsertUploadedDoc(docs, result) {
    const existingIndex = docs.findIndex((doc) => doc.source_id === result.source_id);
    const selected = existingIndex >= 0 ? docs[existingIndex].selected : true;
    const next = { ...result, selected };
    if (existingIndex >= 0) docs.splice(existingIndex, 1, next);
    else docs.push(next);
    return docs;
  },
  isDocumentProcessing(doc) {
    return ["ocr_queued", "ocr_running"].includes(String((doc && doc.status) || ""));
  },
  isDocumentAskable(doc) {
    return ["direct_ready", "rlm_ready"].includes(String((doc && doc.status) || ""));
  },
};
let uploadInFlight = false;
let pollTimer = null;
let pollInFlight = false;

docList = ensureElement(docList, {
  id: "docList",
  tagName: "div",
  className: "doc-list",
  parentSelector: ".side-panel",
  beforeElement: selectionSummary,
});
evidenceBody = ensureEvidenceBody(evidenceBody);

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (uploadInFlight) return;
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  uploadInFlight = true;
  serviceStatus.textContent = "processing";
  askButton.disabled = true;
  uploadButton.disabled = true;
  fileInput.disabled = true;

  try {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/upload", { method: "POST", body: form });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      appState.upsertUploadedDoc(uploadedDocs, result);
    }
    fileInput.value = "";
    renderDocList();
    syncDocumentPolling();
  } catch (error) {
    serviceStatus.textContent = "upload failed";
    answerText.textContent = String(error);
  } finally {
    uploadInFlight = false;
    uploadButton.disabled = false;
    fileInput.disabled = false;
    syncDocumentPolling();
    updateAskState();
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sourceIds = selectedSourceIds();
  if (!sourceIds.length) return;

  askButton.disabled = true;
  answerText.textContent = "思考中...";
  try {
    const payload = {
      session_id: "local_session",
      question: questionInput.value,
      stream: false,
    };
    if (sourceIds.length === 1) {
      payload.source_id = sourceIds[0];
    } else {
      payload.source_ids = sourceIds;
    }
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    renderAnswer(await response.json());
  } catch (error) {
    answerText.textContent = String(error);
  } finally {
    updateAskState();
  }
});

function renderDocList() {
  docList.innerHTML = "";
  if (!uploadedDocs.length) {
    docList.innerHTML = '<div class="empty-state">暂无文档</div>';
    updateAskState();
    return;
  }

  for (const doc of uploadedDocs) {
    const profile = doc.doc_profile;
    const item = document.createElement("label");
    item.className = "doc-item";
    item.innerHTML = `
      <input type="checkbox" ${doc.selected ? "checked" : ""} data-source-id="${escapeHtml(doc.source_id)}" />
      <span class="doc-main">
        <span class="doc-name">${escapeHtml(profile.file_name)}</span>
        <span class="doc-meta">${escapeHtml(doc.status)} · ${profile.page_count} 页 · ${profile.chunk_count} chunks</span>
        <span class="doc-source">${escapeHtml(doc.source_id)}</span>
      </span>
    `;
    item.querySelector("input").addEventListener("change", (event) => {
      doc.selected = event.target.checked;
      updateAskState();
    });
    docList.appendChild(item);
  }
  updateAskState();
}

function selectedSourceIds() {
  return appState.selectedSourceIds(uploadedDocs);
}

function syncDocumentPolling() {
  const hasProcessingDocument = uploadedDocs.some((doc) => appState.isDocumentProcessing(doc));
  if (!hasProcessingDocument) {
    if (pollTimer !== null) {
      globalThis.clearInterval(pollTimer);
      pollTimer = null;
    }
    if (!uploadInFlight) serviceStatus.textContent = "ready";
    return;
  }
  serviceStatus.textContent = "processing";
  if (pollTimer !== null) return;
  pollTimer = globalThis.setInterval(() => {
    void refreshProcessingDocuments();
  }, 1000);
}

async function refreshProcessingDocuments() {
  if (pollInFlight) return;
  const processingDocs = uploadedDocs.filter((doc) => appState.isDocumentProcessing(doc));
  if (!processingDocs.length) {
    syncDocumentPolling();
    return;
  }
  pollInFlight = true;
  try {
    const results = await Promise.all(
      processingDocs.map(async (doc) => {
        const response = await fetch(`/documents/${encodeURIComponent(doc.source_id)}`);
        if (!response.ok) throw new Error(await response.text());
        return response.json();
      }),
    );
    for (const result of results) appState.upsertUploadedDoc(uploadedDocs, result);
    renderDocList();
  } catch (error) {
    serviceStatus.textContent = "status refresh failed";
  } finally {
    pollInFlight = false;
    syncDocumentPolling();
  }
}

function updateAskState() {
  const count = selectedSourceIds().length;
  askButton.disabled = count === 0;
  selectionSummary.textContent = count ? `已选择 ${count} 份文档` : "未选择文档";
}

function renderAnswer(answer) {
  answerText.textContent = answer.answer || "";
  modeTag.textContent = `mode: ${answer.mode}`;
  answerTypeTag.textContent = `answer_type: ${answer.answer_type}`;
  evidenceBody.innerHTML = "";
  for (const evidence of answer.evidences || []) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(evidence.file_name || "")}</td>
      <td>${evidence.page_number}</td>
      <td>${escapeHtml(evidence.quote || "")}</td>
      <td>${escapeHtml(evidence.source_id || "")}</td>
      <td>${escapeHtml(evidence.evidence_id || "")}</td>
      <td>${escapeHtml(evidence.quote_source || "")}</td>
    `;
    evidenceBody.appendChild(row);
  }
  debugText.textContent = JSON.stringify(answer.debug || {}, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function ensureElement(element, options) {
  if (element) return element;
  const created = document.createElement(options.tagName);
  created.id = options.id;
  created.className = options.className;
  const parent = document.querySelector(options.parentSelector) || document.body;
  if (options.beforeElement) {
    parent.insertBefore(created, options.beforeElement);
  } else {
    parent.appendChild(created);
  }
  return created;
}

function ensureEvidenceBody(element) {
  if (element) return element;
  const created = document.createElement("tbody");
  created.id = "evidenceBody";
  const table = document.querySelector("table") || document.body;
  table.appendChild(created);
  return created;
}

renderDocList();
