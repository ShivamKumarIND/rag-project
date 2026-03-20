/* ── DocQA — Frontend Application ──────────────────────────────── */
(function () {
  "use strict";

  // ── Config ──────────────────────────────────────────────────
  const API_BASE = window.location.origin;

  // ── State ───────────────────────────────────────────────────
  let sessionId = crypto.randomUUID();
  let uploadedFile = null;
  let isUploaded = false;

  // ── DOM refs ────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const llmSelect        = $("#llmSelect");
  const dropZone          = $("#dropZone");
  const fileInput         = $("#fileInput");
  const selectedFileEl    = $("#selectedFile");
  const selectedFileName  = $("#selectedFileName");
  const clearFileBtn      = $("#clearFile");
  const uploadBtn         = $("#uploadBtn");
  const uploadSpinner     = $("#uploadSpinner");
  const statusArea        = $("#statusArea");
  const docBadgeSection   = $("#docBadgeSection");
  const docBadgeName      = $("#docBadgeName");
  const chatMessages      = $("#chatMessages");
  const emptyState        = $("#emptyState");
  const questionInput     = $("#questionInput");
  const sendBtn           = $("#sendBtn");
  const sidebarToggle     = $("#sidebarToggle");
  const sidebar           = $("#sidebar");
  const toastContainer    = $("#toastContainer");

  // ── Init ────────────────────────────────────────────────────
  fetchLLMs();
  bindEvents();

  // ── Fetch LLMs ──────────────────────────────────────────────
  async function fetchLLMs() {
    try {
      const res = await fetch(`${API_BASE}/llms`);
      if (!res.ok) throw new Error("Failed to fetch LLMs");
      const data = await res.json();
      llmSelect.innerHTML = "";
      data.llms.forEach((llm) => {
        const opt = document.createElement("option");
        opt.value = llm.id;
        opt.textContent = llm.display_name;
        if (llm.id === data.default) opt.selected = true;
        llmSelect.appendChild(opt);
      });
    } catch (err) {
      showToast("Could not load LLM list. Is the server running?", true);
    }
  }

  // ── Events ──────────────────────────────────────────────────
  function bindEvents() {
    // Sidebar toggle (mobile)
    sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));

    // Drop zone
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      const files = e.dataTransfer.files;
      if (files.length && files[0].type === "application/pdf") {
        setFile(files[0]);
      } else {
        showToast("Please drop a PDF file.", true);
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) setFile(fileInput.files[0]);
    });
    clearFileBtn.addEventListener("click", clearFile);

    // Upload
    uploadBtn.addEventListener("click", handleUpload);

    // Chat
    sendBtn.addEventListener("click", handleSend);
    questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    });

    // Cleanup on unload
    window.addEventListener("beforeunload", () => {
      navigator.sendBeacon(`${API_BASE}/session/${sessionId}`, "");
    });
  }

  // ── File selection ──────────────────────────────────────────
  function setFile(file) {
    uploadedFile = file;
    selectedFileName.textContent = file.name;
    selectedFileEl.style.display = "flex";
    uploadBtn.disabled = false;
    statusArea.innerHTML = "";
  }

  function clearFile() {
    uploadedFile = null;
    fileInput.value = "";
    selectedFileEl.style.display = "none";
    uploadBtn.disabled = true;
  }

  // ── Upload ──────────────────────────────────────────────────
  async function handleUpload() {
    if (!uploadedFile) return;

    uploadBtn.disabled = true;
    uploadSpinner.style.display = "inline-block";
    statusArea.innerHTML = "";

    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("session_id", sessionId);

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        throw new Error("Server error — possibly out of memory. Try a smaller PDF.");
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      setStatus(`Indexed ${data.chunk_count} chunks from "${data.filename}"`, false);
      docBadgeName.textContent = data.filename;
      docBadgeSection.style.display = "block";
      isUploaded = true;
      enableChat();
    } catch (err) {
      setStatus(err.message, true);
      uploadBtn.disabled = false;
    } finally {
      uploadSpinner.style.display = "none";
    }
  }

  // ── Ask ─────────────────────────────────────────────────────
  async function handleSend() {
    const q = questionInput.value.trim();
    if (!q || !isUploaded) return;

    // Hide empty state
    if (emptyState) emptyState.remove();

    appendBubble(q, "user");
    questionInput.value = "";
    sendBtn.disabled = true;
    questionInput.disabled = true;

    // Typing indicator
    const typingEl = appendTyping();

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, session_id: sessionId, llm_id: llmSelect.value }),
      });
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        throw new Error("Server error — please try again.");
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to get answer");

      typingEl.remove();
      appendAnswer(data.answer, data.sources);
    } catch (err) {
      typingEl.remove();
      showToast(err.message, true);
    } finally {
      sendBtn.disabled = false;
      questionInput.disabled = false;
      questionInput.focus();
    }
  }

  // ── Chat helpers ────────────────────────────────────────────
  function enableChat() {
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.placeholder = "Ask a question about your document…";
    questionInput.focus();
  }

  function appendBubble(text, role) {
    const div = document.createElement("div");
    div.className = `chat-bubble ${role}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
  }

  function appendAnswer(answer, sources) {
    const wrapper = document.createElement("div");
    wrapper.className = "chat-bubble assistant";

    const answerP = document.createElement("div");
    answerP.textContent = answer;
    wrapper.appendChild(answerP);

    if (sources && sources.length) {
      const details = document.createElement("details");
      details.className = "sources-details";
      const summary = document.createElement("summary");
      summary.textContent = `Sources (${sources.length})`;
      details.appendChild(summary);

      sources.forEach((src) => {
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `
          <div class="source-page">Page ${src.page !== null && src.page !== "N/A" ? parseInt(src.page) + 1 : "N/A"}</div>
          <div class="source-snippet">${escapeHtml(src.snippet)}</div>
        `;
        details.appendChild(item);
      });

      wrapper.appendChild(details);
    }

    chatMessages.appendChild(wrapper);
    scrollToBottom();
  }

  function appendTyping() {
    const div = document.createElement("div");
    div.className = "chat-bubble assistant typing-indicator";
    div.innerHTML = "<span></span><span></span><span></span>";
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ── Status ──────────────────────────────────────────────────
  function setStatus(msg, isError) {
    statusArea.innerHTML = `<div class="status-msg ${isError ? "status-error" : "status-success"}">${escapeHtml(msg)}</div>`;
  }

  // ── Toast ───────────────────────────────────────────────────
  function showToast(msg, isError = false) {
    const toast = document.createElement("div");
    toast.className = `toast${isError ? " error" : ""}`;
    toast.textContent = msg;
    toastContainer.appendChild(toast);
    setTimeout(() => { toast.style.opacity = "0"; setTimeout(() => toast.remove(), 300); }, 4000);
  }

  // ── Utils ───────────────────────────────────────────────────
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
