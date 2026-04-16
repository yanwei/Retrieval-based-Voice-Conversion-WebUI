const state = {
  singleJobId: null,
  longJobId: null,
  options: null,
  jobTimers: {},
};

function $(id) {
  return document.getElementById(id);
}

function setRangePair(rangeId, valueId) {
  const range = $(rangeId);
  const value = $(valueId);
  const sync = (from, to) => {
    to.value = from.value;
  };
  range.addEventListener("input", () => sync(range, value));
  value.addEventListener("input", () => sync(value, range));
}

function setAudioPreview(fileInputId, audioId) {
  $(fileInputId).addEventListener("change", (event) => {
    const file = event.target.files[0];
    $(audioId).src = file ? URL.createObjectURL(file) : "";
  });
}

function fillSelect(select, values, picked) {
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "留空 / 自动匹配";
  select.appendChild(empty);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === picked) option.selected = true;
    select.appendChild(option);
  });
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

async function loadOptions() {
  const data = await fetchJson("/api/options");
  state.options = data;
  fillSelect($("single-model"), data.models, data.models[0] || "");
  fillSelect($("male-model"), data.models, data.default_male || data.models[0] || "");
  fillSelect($("female-model"), data.models, data.default_female || data.models[0] || "");
  fillSelect($("single-index"), data.indices, "");
  fillSelect($("male-index"), data.indices, "");
  fillSelect($("female-index"), data.indices, "");
  await refreshIndexForModel("single-model", "single-index");
  await refreshIndexForModel("male-model", "male-index");
  await refreshIndexForModel("female-model", "female-index");
}

async function refreshIndexForModel(modelSelectId, indexSelectId) {
  const model = $(modelSelectId).value;
  const result = await fetchJson(`/api/default-index?model=${encodeURIComponent(model)}`);
  const select = $(indexSelectId);
  fillSelect(select, state.options.indices, result.index || "");
}

function setProgress(prefix, progress, message, log, resultAudio, downloads, extraAudio, etaText) {
  $(`${prefix}-progress-bar`).style.width = `${Math.round((progress || 0) * 100)}%`;
  const parts = [`${Math.round((progress || 0) * 100)}%`, message || ""];
  if (etaText) parts.push(etaText);
  $(`${prefix}-status`).textContent = parts.filter(Boolean).join(" | ");
  if (log) {
    $(`${prefix}-log`).textContent = log.join("\n");
  }
  if (resultAudio && $(`${prefix}-result`)) {
    $(`${prefix}-result`).src = `/download?path=${encodeURIComponent(resultAudio)}`;
  }
  if (extraAudio && $(`${prefix}-vocals`)) {
    $(`${prefix}-vocals`).src = `/download?path=${encodeURIComponent(extraAudio)}`;
  }
  const links = $(`${prefix}-downloads`);
  if (links) {
    links.innerHTML = "";
    (downloads || []).forEach((path) => {
      const a = document.createElement("a");
      a.href = `/download?path=${encodeURIComponent(path)}`;
      a.textContent = `下载 ${path.split("/").slice(-1)[0]}`;
      a.target = "_blank";
      links.appendChild(a);
    });
  }
}

async function pollJob(jobId, prefix) {
  try {
    const data = await fetchJson(`/api/jobs/${jobId}`);
    if (!state.jobTimers[jobId]) {
      state.jobTimers[jobId] = {
        startedAt: data.started_at || Date.now() / 1000,
      };
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
      prefix,
      data.progress,
      data.message,
      data.log,
      data.result_audio,
      data.downloads,
      data.converted_vocals,
      etaText
    );
    if (data.status === "done") return;
    if (data.status === "error") {
      $(`${prefix}-log`).textContent = `${(data.log || []).join("\n")}\n\n${data.error || ""}`;
      throw new Error(data.error || "处理失败");
    }
    setTimeout(() => pollJob(jobId, prefix).catch((err) => {
      $(`${prefix}-status`).textContent = `Error: ${err.message}`;
    }), 1200);
  } catch (error) {
    const status = $(`${prefix}-status`);
    status.textContent = `轮询重试中: ${error.message}`;
    setTimeout(() => pollJob(jobId, prefix).catch((err) => {
      $(`${prefix}-status`).textContent = `Error: ${err.message}`;
    }), 2000);
  }
}

async function submitSingle() {
  const file = $("single-audio").files[0];
  if (!file) {
    alert("请先选择原始音频文件");
    return;
  }
  const form = new FormData();
  form.append("audio", file);
  form.append("model_name", $("single-model").value);
  form.append("index_path", $("single-index").value);
  form.append("use_index", $("single-use-index").checked ? "true" : "false");
  form.append("f0_up_key", $("single-f0-up-key").value);
  form.append("f0_method", $("single-f0-method").value);
  form.append("index_rate", $("single-index-rate").value);
  form.append("protect", $("single-protect").value);
  form.append("rms_mix_rate", $("single-rms").value);
  form.append("filter_radius", $("single-filter-radius").value);
  form.append("resample_sr", $("single-resample-sr").value);
  setProgress("single", 0, "任务已提交", [], "", []);
  const result = await fetchJson("/api/jobs/single", { method: "POST", body: form });
  state.singleJobId = result.job_id;
  state.jobTimers[result.job_id] = { startedAt: Date.now() / 1000 };
  pollJob(result.job_id, "single").catch((err) => {
    $("single-status").textContent = `Error: ${err.message}`;
  });
}

async function submitLong() {
  const file = $("long-audio").files[0];
  if (!file) {
    alert("请先选择原始音频文件");
    return;
  }
  const form = new FormData();
  form.append("audio", file);
  form.append("male_model", $("male-model").value);
  form.append("male_index", $("male-index").value);
  form.append("male_f0_up_key", $("male-f0-up-key").value);
  form.append("male_index_rate", $("male-index-rate").value);
  form.append("male_protect", $("male-protect").value);
  form.append("male_rms_mix_rate", $("male-rms").value);
  form.append("female_model", $("female-model").value);
  form.append("female_index", $("female-index").value);
  form.append("female_f0_up_key", $("female-f0-up-key").value);
  form.append("female_index_rate", $("female-index-rate").value);
  form.append("female_protect", $("female-protect").value);
  form.append("female_rms_mix_rate", $("female-rms").value);
  form.append("f0_method", $("long-f0-method").value);
  form.append("filter_radius", $("long-filter-radius").value);
  form.append("resample_sr", $("long-resample-sr").value);
  form.append("remix", $("long-remix").checked ? "true" : "false");
  form.append("reading_mode", $("long-reading-mode").checked ? "true" : "false");
  form.append("speaker_embedding", $("long-speaker-embedding").checked ? "true" : "false");
  setProgress("long", 0, "任务已提交", [], "", []);
  const result = await fetchJson("/api/jobs/long", { method: "POST", body: form });
  state.longJobId = result.job_id;
  state.jobTimers[result.job_id] = { startedAt: Date.now() / 1000 };
  pollJob(result.job_id, "long").catch((err) => {
    $("long-status").textContent = `Error: ${err.message}`;
  });
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  [
    ["single-f0-up-key", "single-f0-up-key-value"],
    ["single-index-rate", "single-index-rate-value"],
    ["single-protect", "single-protect-value"],
    ["single-rms", "single-rms-value"],
    ["single-filter-radius", "single-filter-radius-value"],
    ["single-resample-sr", "single-resample-sr-value"],
    ["male-f0-up-key", "male-f0-up-key-value"],
    ["male-index-rate", "male-index-rate-value"],
    ["male-protect", "male-protect-value"],
    ["male-rms", "male-rms-value"],
    ["female-f0-up-key", "female-f0-up-key-value"],
    ["female-index-rate", "female-index-rate-value"],
    ["female-protect", "female-protect-value"],
    ["female-rms", "female-rms-value"],
    ["long-filter-radius", "long-filter-radius-value"],
    ["long-resample-sr", "long-resample-sr-value"],
  ].forEach(([rangeId, valueId]) => setRangePair(rangeId, valueId));

  setAudioPreview("single-audio", "single-original");
  setAudioPreview("long-audio", "long-original");

  await loadOptions();
  $("single-model").addEventListener("change", () => refreshIndexForModel("single-model", "single-index"));
  $("male-model").addEventListener("change", () => refreshIndexForModel("male-model", "male-index"));
  $("female-model").addEventListener("change", () => refreshIndexForModel("female-model", "female-index"));
  $("single-submit").addEventListener("click", () => submitSingle().catch((err) => {
    $("single-status").textContent = `Error: ${err.message}`;
  }));
  $("long-submit").addEventListener("click", () => submitLong().catch((err) => {
    $("long-status").textContent = `Error: ${err.message}`;
  }));
});
