(function (root) {
  function docDisplayKey(doc) {
    return String(
      (doc && doc.doc_profile && doc.doc_profile.file_name) ||
        (doc && doc.manifest && doc.manifest.file_name) ||
        (doc && doc.source_id) ||
        ""
    );
  }

  function upsertUploadedDoc(docs, result) {
    const key = docDisplayKey(result);
    const existingIndex = docs.findIndex((doc) => docDisplayKey(doc) === key);
    const selected = existingIndex >= 0 ? docs[existingIndex].selected : true;
    const next = { ...result, selected };
    if (existingIndex >= 0) {
      docs.splice(existingIndex, 1, next);
    } else {
      docs.push(next);
    }
    return docs;
  }

  function selectedSourceIds(docs) {
    return docs.filter((doc) => doc.selected && isDocumentAskable(doc)).map((doc) => doc.source_id);
  }

  function isDocumentProcessing(doc) {
    return ["ocr_queued", "ocr_running"].includes(String((doc && doc.status) || ""));
  }

  function isDocumentAskable(doc) {
    return ["direct_ready", "rlm_ready"].includes(String((doc && doc.status) || ""));
  }

  const api = {
    docDisplayKey,
    isDocumentAskable,
    isDocumentProcessing,
    selectedSourceIds,
    upsertUploadedDoc,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.RLMQAState = api;
})(typeof window !== "undefined" ? window : globalThis);
