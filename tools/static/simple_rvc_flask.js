const state = {
  autoJobId: null,
  options: null,
  jobTimers: {},
};

function $(id) {
  return document.getElementById(id);
}

function prettyJson(value) {
  if (!value) return "暂无";
  if (typeof value === "object" && Object.keys(value).length === 0) return "暂无";
  return JSON.stringify(value, null, 2);
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  const rounded = Math.round(seconds);
  const mins = Math.floor(rounded / 60);
  const secs = rounded % 60;
  if (mins <= 0) return `预计剩余 ${secs}s`;
  return `预计剩余 ${mins}m ${secs}s`;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function setAudioPreview(fileInputId, audioId) {
  $(fileInputId).addEventListener("change", (event) => {
    const file = event.target.files[0];
    $(audioId).src = file ? URL.createObjectURL(file) : "";
  });
}

async function loadOptions() {
  const data = await fetchJson("/api/options");
  state.options = data;
  $("auto-thresholds").textContent = prettyJson({
    active_thresholds: data.active_thresholds || {},
    note: "这些阈值来自全量音频分析结果，用于黑盒自动决策。",
  });
  $("auto-metadata").textContent = prettyJson({
    note: "这里会显示 analysis / selected_plan / quality_gate / review / stage_summaries。",
  });
}

function renderWarning(metadata) {
  const node = $("auto-warning");
  const result = metadata && metadata.result_metadata ? metadata.result_metadata : metadata;
  if (!result) {
    node.hidden = true;
    node.textContent = "";
    return;
  }

  const qualityGate = result.quality_gate || {};
  const review = result.review || {};
  const pieces = [];
  if (result.status === "fallback") {
    pieces.push("本次任务触发了回退，输出可能是原音或保守结果。");
  }
  if (qualityGate.fallback_used) {
    pieces.push(`quality_gate: ${qualityGate.fallback_reason || "fallback_used"}`);
  }
  if (Array.isArray(qualityGate.warnings) && qualityGate.warnings.length) {
    pieces.push(`warnings: ${qualityGate.warnings.join(", ")}`);
  }
  if (review.needs_review) {
    pieces.push(`review: ${(review.reasons || []).join(", ") || "needs_review"}`);
  }

  if (pieces.length === 0) {
    node.hidden = true;
    node.textContent = "";
    return;
  }
  node.hidden = false;
  node.textContent = pieces.join(" ");
}

function setProgress(progress, message, log, resultAudio, downloads, etaText, metadata) {
  $("auto-progress-bar").style.width = `${Math.round((progress || 0) * 100)}%`;
  const parts = [`${Math.round((progress || 0) * 100)}%`, message || ""];
  if (etaText) parts.push(etaText);
  $("auto-status").textContent = parts.filter(Boolean).join(" | ");
  $("auto-log").textContent = (log || []).join("\n");

  if (resultAudio) {
    $("auto-result").src = `/download?path=${encodeURIComponent(resultAudio)}`;
  }

  const links = $("auto-downloads");
  links.innerHTML = "";
  (downloads || []).forEach((path) => {
    const a = document.createElement("a");
    a.href = `/download?path=${encodeURIComponent(path)}`;
    a.textContent = `下载 ${path.split("/").slice(-1)[0]}`;
    a.target = "_blank";
    links.appendChild(a);
  });

  $("auto-metadata").textContent = prettyJson(metadata);
  renderWarning(metadata);
}

async function pollJob(jobId) {
  try {
    const data = await fetchJson(`/api/jobs/${jobId}`);
    if (!state.jobTimers[jobId]) {
      state.jobTimers[jobId] = { startedAt: data.started_at || Date.now() / 1000 };
    }
    let etaText = "";
    const startedAt = state.jobTimers[jobId].startedAt;
    const now = Date.now() / 1000;
    if (data.progress > 0.05 && data.progress < 0.995 && startedAt) {
      const elapsed = now - startedAt;
      const etaSeconds = elapsed * (1 - data.progress) / data.progress;
      etaText = formatEta(etaSeconds);
    }

    setProgress(
      data.progress,
      data.message,
      data.log,
      data.result_audio,
      data.downloads,
      etaText,
      data.metadata
    );

    if (data.status === "done") return;
    if (data.status === "error") {
      $("auto-log").textContent = `${(data.log || []).join("\n")}\n\n${data.error || ""}`;
      throw new Error(data.error || "处理失败");
    }

    setTimeout(() => pollJob(jobId).catch((err) => {
      $("auto-status").textContent = `Error: ${err.message}`;
    }), 1200);
  } catch (error) {
    $("auto-status").textContent = `轮询重试中: ${error.message}`;
    setTimeout(() => pollJob(jobId).catch((err) => {
      $("auto-status").textContent = `Error: ${err.message}`;
    }), 2000);
  }
}

async function submitAuto() {
  const file = $("auto-audio").files[0];
  if (!file) {
    alert("请先选择原始音频文件");
    return;
  }
  const form = new FormData();
  form.append("audio", file);
  form.append("profile", $("auto-profile").value || "default");
  setProgress(0, "任务已提交", [], "", [], "", {});
  const result = await fetchJson("/api/jobs/auto", { method: "POST", body: form });
  state.autoJobId = result.job_id;
  state.jobTimers[result.job_id] = { startedAt: Date.now() / 1000 };
  pollJob(result.job_id).catch((err) => {
    $("auto-status").textContent = `Error: ${err.message}`;
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  setAudioPreview("auto-audio", "auto-original");
  await loadOptions();
  $("auto-submit").addEventListener("click", () => submitAuto().catch((err) => {
    $("auto-status").textContent = `Error: ${err.message}`;
  }));
});
