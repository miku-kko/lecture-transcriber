// ============================================================
// Lecture Transcriber — Frontend Application
// ============================================================

const SPEAKER_LABELS = {
    lecturer: "Wykladowca",
    student: "Student",
    unknown: "Nieznany",
};

const SPEAKER_EMOJI = {
    lecturer: "\ud83c\udf93",
    student: "\ud83d\ude4b",
    unknown: "\u2753",
};

// --- State ---
const state = {
    ws: null,
    recording: false,
    mode: "automatic",
    segments: [],         // Array of finalized segments
    corrections: {},      // segment_id -> correction data
    selectedSegmentId: null,
    speakers: new Set(),
};

// --- DOM Elements ---
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    btnRecord: $("#btn-record"),
    btnStop: $("#btn-stop"),
    sessionTitle: $("#session-title"),
    recordingIndicator: $("#recording-indicator"),
    transcriptContainer: $("#transcript-container"),
    transcriptContent: $("#transcript-content"),
    interimText: $("#interim-text"),
    emptyState: $("#empty-state"),
    correctionDetails: $("#correction-details"),
    ragQuery: $("#rag-query"),
    btnRagSearch: $("#btn-rag-search"),
    ragResults: $("#rag-results"),
    btnRefreshLectures: $("#btn-refresh-lectures"),
    lectureList: $("#lecture-list"),
    speakerMap: $("#speaker-map"),
    keytermsInput: $("#keyterms-input"),
    btnSaveKeyterms: $("#btn-save-keyterms"),
    statusOllama: $("#status-ollama"),
    statusDeepgram: $("#status-deepgram"),
    toast: $("#toast"),
    // Audio gain & level
    gainSlider: $("#gain-slider"),
    gainValue: $("#gain-value"),
    agcCheckbox: $("#agc-checkbox"),
    levelBarRms: $("#level-bar-rms"),
    levelBarPeak: $("#level-bar-peak"),
    levelDb: $("#level-db"),
};

// ============================================================
// WebSocket Connection
// ============================================================

function connectWS() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log("WebSocket connected");
        showToast("Polaczono z serwerem", "success");
        pollStatus();
        sendWS("cmd_list_lectures");
    };

    state.ws.onclose = () => {
        console.log("WebSocket disconnected");
        showToast("Rozlaczono — ponowne laczenie...", "error");
        setTimeout(connectWS, 3000);
    };

    state.ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };

    state.ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleWSMessage(msg);
        } catch (e) {
            console.error("Failed to parse WS message:", e);
        }
    };
}

function sendWS(type, payload = {}) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            type,
            payload,
            timestamp: Date.now() / 1000,
        }));
    }
}

// ============================================================
// Message Handlers
// ============================================================

function handleWSMessage(msg) {
    console.log("[WS]", msg.type, msg.payload?.segment?.text || "");
    const handlers = {
        transcript_interim: handleTranscriptInterim,
        transcript_final: handleTranscriptFinal,
        correction_result: handleCorrectionResult,
        session_started: handleSessionStarted,
        session_stopped: handleSessionStopped,
        rag_result: handleRagResult,
        lecture_list: handleLectureList,
        error: handleError,
        status: handleStatus,
        audio_level: handleAudioLevel,
    };

    const handler = handlers[msg.type];
    if (handler) {
        handler(msg.payload);
    } else {
        console.warn("Unknown message type:", msg.type);
    }
}

function handleTranscriptInterim(payload) {
    const { segment } = payload;
    const label = SPEAKER_LABELS[segment.speaker_role] || `Mowca ${segment.speaker}`;
    dom.interimText.textContent = `${label}: ${segment.text}`;
    scrollToBottom();
}

function handleTranscriptFinal(payload) {
    const { segment, index } = payload;
    dom.interimText.textContent = "";

    state.segments.push(segment);
    trackSpeaker(segment.speaker, segment.speaker_role);
    renderSegment(segment);
    scrollToBottom();
}

function handleCorrectionResult(payload) {
    const { segment_id, corrected_text, items, model, processing_ms, auto_applied } = payload;
    state.corrections[segment_id] = { corrected_text, items, model, processing_ms, auto_applied };

    // Batched correction: segment_id contains multiple IDs joined with "|"
    const segmentIds = segment_id.split("|");
    const isBatch = segmentIds.length > 1;

    if (isBatch && corrected_text) {
        // Replace all batched segments with one corrected block
        const firstEl = dom.transcriptContent.querySelector(
            `.segment[data-segment-id="${segmentIds[0]}"]`
        );
        if (firstEl) {
            // Create corrected block
            const correctedDiv = document.createElement("div");
            correctedDiv.className = "segment corrected-batch";
            correctedDiv.dataset.segmentId = segment_id;
            correctedDiv.innerHTML = `
                <div class="correction-badge">Skorygowano (${segmentIds.length} segmentow)</div>
                <div class="text corrected-text" data-segment-id="${segment_id}">${escapeHtml(corrected_text)}</div>
            `;

            // Insert before first segment
            firstEl.parentNode.insertBefore(correctedDiv, firstEl);

            // Hide original segments
            segmentIds.forEach((id) => {
                const el = dom.transcriptContent.querySelector(
                    `.segment[data-segment-id="${id}"]`
                );
                if (el) el.style.display = "none";
            });
        }

        // Highlight corrections in the new block
        if (items && items.length > 0) {
            highlightCorrections(segment_id, items);
        }
    } else {
        // Single segment correction (legacy behavior)
        const textEl = dom.transcriptContent.querySelector(
            `.text[data-segment-id="${segment_id}"]`
        );

        if (auto_applied && corrected_text) {
            if (textEl) {
                textEl.textContent = corrected_text;
            }
            const seg = state.segments.find((s) => s.segment_id === segment_id);
            if (seg) {
                seg.text = corrected_text;
            }
        }

        if (items && items.length > 0) {
            highlightCorrections(segment_id, items);
        }
    }
}

function handleSessionStarted(payload) {
    state.recording = true;
    state.segments = [];
    state.corrections = {};
    state.speakers.clear();

    dom.btnRecord.disabled = true;
    dom.btnStop.disabled = false;
    dom.recordingIndicator.classList.remove("hidden");
    dom.emptyState.style.display = "none";
    dom.transcriptContainer.classList.add("active");
    dom.transcriptContent.innerHTML = "";
    dom.interimText.textContent = "";

    showToast(`Nagrywanie rozpoczete: ${payload.title}`, "success");
}

function handleSessionStopped(payload) {
    state.recording = false;

    dom.btnRecord.disabled = false;
    dom.btnStop.disabled = true;
    dom.recordingIndicator.classList.add("hidden");

    const msg = payload.file_path
        ? `Sesja zapisana: ${payload.file_path}`
        : "Sesja zakonczona";
    showToast(msg, "success");
}

function handleRagResult(payload) {
    const { results } = payload;
    dom.ragResults.innerHTML = "";

    if (!results || results.length === 0) {
        dom.ragResults.innerHTML = '<p class="placeholder" style="color: var(--text-muted); font-size: 0.8rem;">Brak wynikow</p>';
        return;
    }

    results.forEach((r) => {
        const item = document.createElement("div");
        item.className = "result-item";
        const title = r.metadata?.title || r.metadata?.lecture_id || "Wyklad";
        const snippet = r.text ? r.text.substring(0, 150) + "..." : "";
        const distance = r.distance ? `(${(1 - r.distance).toFixed(2)} dopasowanie)` : "";
        item.innerHTML = `
            <div class="title">${escapeHtml(title)}</div>
            <div class="meta">${distance}</div>
            <div class="snippet">${escapeHtml(snippet)}</div>
        `;
        dom.ragResults.appendChild(item);
    });
}

function handleLectureList(payload) {
    const { lectures } = payload;
    dom.lectureList.innerHTML = "";

    if (!lectures || lectures.length === 0) {
        dom.lectureList.innerHTML = '<p class="placeholder" style="color: var(--text-muted); font-size: 0.8rem;">Brak wykladow</p>';
        return;
    }

    lectures.forEach((l) => {
        const item = document.createElement("div");
        item.className = "result-item";
        const date = new Date(l.modified * 1000).toLocaleDateString("pl-PL");
        const sizeKB = Math.round(l.size_bytes / 1024);
        item.innerHTML = `
            <div class="title">${escapeHtml(l.filename)}</div>
            <div class="meta">${date} | ${sizeKB} KB</div>
        `;
        item.addEventListener("click", () => openLecture(l.path));
        dom.lectureList.appendChild(item);
    });
}

function handleError(payload) {
    console.error("Server error:", payload.error);
    showToast(`Blad: ${payload.error}`, "error");
}

function handleStatus(payload) {
    if (payload.ollama_available !== undefined) {
        dom.statusOllama.textContent = `Ollama: ${payload.ollama_available ? "OK" : "OFF"}`;
        dom.statusOllama.className = `status-badge ${payload.ollama_available ? "ok" : "error"}`;
    }
    if (payload.deepgram_configured !== undefined) {
        dom.statusDeepgram.textContent = `DG: ${payload.deepgram_configured ? "OK" : "OFF"}`;
        dom.statusDeepgram.className = `status-badge ${payload.deepgram_configured ? "ok" : "error"}`;
    }
}

function handleAudioLevel(payload) {
    const { rms, peak, gain, effective_gain } = payload;

    // Update level meter bars
    const rmsPercent = Math.min(rms * 100 * 3, 100); // scale up for visibility
    const peakPercent = Math.min(peak * 100 * 2, 100);
    dom.levelBarRms.style.width = `${rmsPercent}%`;
    dom.levelBarPeak.style.width = `${peakPercent}%`;

    // Color: green if ok, red if clipping
    if (rmsPercent > 85) {
        dom.levelBarRms.classList.add("hot");
    } else {
        dom.levelBarRms.classList.remove("hot");
    }

    // Show dB approximation
    const db = rms > 0 ? Math.round(20 * Math.log10(rms)) : -60;
    dom.levelDb.textContent = `${db}dB`;
}

// ============================================================
// Rendering
// ============================================================

function renderSegment(segment) {
    const div = document.createElement("div");
    div.className = "segment";
    div.dataset.segmentId = segment.segment_id;
    div.setAttribute("data-segment-id", segment.segment_id);

    const speakerClass = segment.speaker_role || "unknown";
    const label = SPEAKER_LABELS[segment.speaker_role] || `Mowca ${segment.speaker}`;
    const emoji = SPEAKER_EMOJI[segment.speaker_role] || "";

    // Check if we need a speaker label (new speaker or first segment)
    const prevSegment = state.segments.length > 1
        ? state.segments[state.segments.length - 2]
        : null;
    const showLabel = !prevSegment || prevSegment.speaker !== segment.speaker;

    let html = "";
    if (showLabel) {
        html += `<div class="speaker-label ${speakerClass}">${emoji} ${label}</div>`;
    }
    html += `<div class="text" data-segment-id="${segment.segment_id}">${escapeHtml(segment.text)}</div>`;

    div.innerHTML = html;
    dom.transcriptContent.appendChild(div);
}

function highlightCorrections(segmentId, items) {
    const textEl = dom.transcriptContent.querySelector(
        `.text[data-segment-id="${segmentId}"]`
    );
    if (!textEl) return;

    const displayedText = textEl.textContent;
    // For batch corrections the displayed text is corrected_text,
    // so we search for item.suggested (the corrected fragment).
    // For single-segment corrections the displayed text is the original,
    // so we search for item.original.
    const isBatch = segmentId.includes("|");

    // Find positions of all corrections and sort by position
    const positioned = items
        .map((item) => {
            const needle = isBatch ? item.suggested : item.original;
            const idx = displayedText.indexOf(needle);
            return idx >= 0 ? { ...item, idx, matchLen: needle.length } : null;
        })
        .filter(Boolean)
        .sort((a, b) => a.idx - b.idx);

    if (positioned.length === 0) return;

    // Build HTML by walking through the displayed text
    let html = "";
    let cursor = 0;

    for (const item of positioned) {
        if (item.idx < cursor) continue; // overlapping, skip

        // Add escaped text before this correction
        html += escapeHtml(displayedText.substring(cursor, item.idx));

        // Add the highlighted correction
        const matchText = escapeHtml(displayedText.substring(item.idx, item.idx + item.matchLen));
        html += `<span class="correction-highlight ${escapeAttr(item.severity)}" data-segment-id="${escapeAttr(segmentId)}" data-original="${escapeAttr(item.original)}" data-suggested="${escapeAttr(item.suggested)}" data-explanation="${escapeAttr(item.explanation)}" data-type="${escapeAttr(item.type)}" data-severity="${escapeAttr(item.severity)}">${matchText}</span>`;

        cursor = item.idx + item.matchLen;
    }

    // Add remaining text
    html += escapeHtml(displayedText.substring(cursor));

    textEl.innerHTML = html;

    // Attach click handlers
    textEl.querySelectorAll(".correction-highlight").forEach((el) => {
        el.addEventListener("click", () => showCorrectionDetail(el));
    });
}

function showCorrectionDetail(el) {
    const segmentId = el.dataset.segmentId;
    const original = el.dataset.original;
    const suggested = el.dataset.suggested;
    const explanation = el.dataset.explanation;
    const type = el.dataset.type;
    const severity = el.dataset.severity;

    state.selectedSegmentId = segmentId;

    const severityIcon = { error: "\ud83d\udd34", warning: "\ud83d\udfe1", info: "\ud83d\udd35" }[severity] || "";

    dom.correctionDetails.innerHTML = `
        <div class="correction-item">
            <div><span class="type-badge">${type}</span> ${severityIcon}</div>
            <div style="margin-top: 0.5rem;">
                <span class="original">${escapeHtml(original)}</span>
                &rarr;
                <span class="suggested">${escapeHtml(suggested)}</span>
            </div>
            <div class="explanation">${escapeHtml(explanation)}</div>
        </div>
    `;
}

function trackSpeaker(speakerId, role) {
    if (!state.speakers.has(speakerId)) {
        state.speakers.add(speakerId);
        renderSpeakerMap();
    }
}

function renderSpeakerMap() {
    dom.speakerMap.innerHTML = "";
    state.speakers.forEach((id) => {
        const entry = document.createElement("div");
        entry.className = "speaker-entry";
        entry.innerHTML = `
            <span class="speaker-id">Mowca #${id}</span>
            <select data-speaker-id="${id}">
                <option value="lecturer">Wykladowca</option>
                <option value="student">Student</option>
                <option value="unknown">Nieznany</option>
            </select>
        `;
        const select = entry.querySelector("select");
        select.addEventListener("change", () => {
            sendWS("cmd_set_speaker_role", {
                speaker_id: id,
                role: select.value,
            });
        });
        dom.speakerMap.appendChild(entry);
    });
}

// ============================================================
// Event Listeners
// ============================================================

dom.btnRecord.addEventListener("click", () => {
    const title = dom.sessionTitle.value.trim();
    sendWS("cmd_start_recording", { title, metadata: {} });
});

dom.btnStop.addEventListener("click", () => {
    sendWS("cmd_stop_recording");
});

// Mode is always automatic
state.mode = "automatic";

dom.btnRagSearch.addEventListener("click", () => {
    const query = dom.ragQuery.value.trim();
    if (query) {
        sendWS("cmd_rag_query", { query });
    }
});

dom.ragQuery.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        dom.btnRagSearch.click();
    }
});

dom.btnRefreshLectures.addEventListener("click", () => {
    sendWS("cmd_list_lectures");
});

dom.btnSaveKeyterms.addEventListener("click", () => {
    const raw = dom.keytermsInput.value.trim();
    const keyterms = raw.split("\n").map((t) => t.trim()).filter((t) => t.length > 0);
    sendWS("cmd_set_keyterms", { keyterms });
    showToast(`Zapisano ${keyterms.length} terminow`, "success");
});

dom.sessionTitle.addEventListener("change", () => {
    const title = dom.sessionTitle.value.trim();
    if (title && state.recording) {
        sendWS("cmd_set_title", { title });
    }
});

// --- Gain / AGC controls ---
dom.gainSlider.addEventListener("input", () => {
    const gain = parseFloat(dom.gainSlider.value);
    dom.gainValue.textContent = `${gain.toFixed(1)}x`;
    sendWS("cmd_set_gain", { gain, agc: dom.agcCheckbox.checked });
});

dom.agcCheckbox.addEventListener("change", () => {
    const gain = parseFloat(dom.gainSlider.value);
    sendWS("cmd_set_gain", { gain, agc: dom.agcCheckbox.checked });
});

// ============================================================
// Utilities
// ============================================================

async function openLecture(path) {
    try {
        const resp = await fetch(`/api/lecture?path=${encodeURIComponent(path)}`);
        if (!resp.ok) {
            showToast("Nie udalo sie otworzyc pliku", "error");
            return;
        }
        const text = await resp.text();

        // Show in center panel
        dom.emptyState.style.display = "none";
        dom.transcriptContainer.classList.add("active");
        dom.transcriptContent.innerHTML = renderMarkdown(text);
        dom.interimText.textContent = "";

        showToast("Wyklad otwarty", "success");
    } catch (e) {
        console.error("Failed to open lecture:", e);
        showToast("Blad odczytu pliku", "error");
    }
}

function renderMarkdown(md) {
    // Strip YAML front matter
    const stripped = md.replace(/^---[\s\S]*?---\n*/m, "");

    // Simple markdown to HTML
    let html = escapeHtml(stripped);

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3 style="color:var(--accent-blue);margin:1rem 0 0.5rem;">$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2 style="color:var(--accent-blue);margin:1.2rem 0 0.5rem;">$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1 style="color:var(--accent-blue);margin:1.5rem 0 0.5rem;">$1</h1>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic / strikethrough
    html = html.replace(/~~(.+?)~~/g, '<del style="color:var(--accent-red);">$1</del>');
    html = html.replace(/_(.+?)_/g, '<em style="color:var(--text-secondary);">$1</em>');

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote style="border-left:3px solid var(--border);padding-left:0.8rem;margin:0.3rem 0;color:var(--text-primary);">$1</blockquote>');

    // List items
    html = html.replace(/^- (.+)$/gm, '<div style="padding-left:1rem;margin:0.2rem 0;">• $1</div>');

    // Table (simple — just hide it, the info is in headers)
    html = html.replace(/\|[^\n]+\|/g, '');

    // Line breaks
    html = html.replace(/\n\n/g, '<div style="margin:0.5rem 0;"></div>');
    html = html.replace(/\n/g, '<br>');

    return `<div class="lecture-view" style="line-height:1.7;font-size:0.9rem;">${html}</div>`;
}

let _scrollRafPending = false;
function scrollToBottom() {
    if (_scrollRafPending) return;
    _scrollRafPending = true;
    requestAnimationFrame(() => {
        dom.transcriptContainer.scrollTop = dom.transcriptContainer.scrollHeight;
        _scrollRafPending = false;
    });
}

const _escapeDiv = document.createElement("div");
function escapeHtml(text) {
    _escapeDiv.textContent = text;
    return _escapeDiv.innerHTML;
}

function escapeAttr(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function showToast(message, type = "success") {
    dom.toast.textContent = message;
    dom.toast.className = `toast ${type}`;
    setTimeout(() => {
        dom.toast.classList.add("hidden");
    }, 4000);
}

// ============================================================
// Status Polling
// ============================================================

async function pollStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();

        dom.statusOllama.textContent = `Ollama: ${data.ollama_available ? "OK" : "OFF"}`;
        dom.statusOllama.className = `status-badge ${data.ollama_available ? "ok" : "error"}`;

        dom.statusDeepgram.textContent = `DG: ${data.deepgram_configured ? "OK" : "OFF"}`;
        dom.statusDeepgram.className = `status-badge ${data.deepgram_configured ? "ok" : "error"}`;

        // Sync gain/AGC state
        if (data.gain !== undefined) {
            dom.gainSlider.value = data.gain;
            dom.gainValue.textContent = `${data.gain.toFixed(1)}x`;
        }
        if (data.agc_enabled !== undefined) {
            dom.agcCheckbox.checked = data.agc_enabled;
        }

        // Sync recording state
        if (data.recording && !state.recording) {
            state.recording = true;
            dom.btnRecord.disabled = true;
            dom.btnStop.disabled = false;
            dom.recordingIndicator.classList.remove("hidden");
            dom.emptyState.style.display = "none";
            dom.transcriptContainer.classList.add("active");
        } else if (!data.recording && state.recording) {
            state.recording = false;
            dom.btnRecord.disabled = false;
            dom.btnStop.disabled = true;
            dom.recordingIndicator.classList.add("hidden");
        }
    } catch (e) {
        dom.statusOllama.textContent = "Ollama: --";
        dom.statusDeepgram.textContent = "DG: --";
    }
}

// ============================================================
// Init
// ============================================================

function init() {
    connectWS();
    pollStatus();
    setInterval(pollStatus, 30000);

    // Load lecture list on start
    setTimeout(() => sendWS("cmd_list_lectures"), 1000);
}

init();
