const state = { items: [], index: 0 };

const player = document.getElementById("player");
const outputPlayer = document.getElementById("outputPlayer");
const outputBlock = document.getElementById("outputBlock");
const sampleTitle = document.getElementById("sampleTitle");
const samplePath = document.getElementById("samplePath");
const sampleMeta = document.getElementById("sampleMeta");
const sampleSummary = document.getElementById("sampleSummary");
const outputSummary = document.getElementById("outputSummary");
const counter = document.getElementById("counter");
const speakerPatternSelect = document.getElementById("speaker_pattern");
const musicPatternSelect = document.getElementById("music_pattern");
const modeHint = document.getElementById("modeHint");

function basename(path) {
  if (!path) return "";
  const parts = String(path).split("/");
  return parts[parts.length - 1] || path;
}

function textOrDash(value) {
  return value ? String(value) : "-";
}

function renderSummary(item) {
  const rows = [
    ["sample_id", textOrDash(item.sample_id)],
    ["source", textOrDash(item.source)],
    ["book_id", textOrDash(item.book_id)],
    ["quality_label", textOrDash(item.quality_label)],
    ["voice_age", textOrDash(item.voice_age)],
    ["system_processing_guess", textOrDash(item.expected_processing_mode)],
  ];
  sampleSummary.innerHTML = rows
    .map(([label, value]) => `<div><strong>${label}</strong>: <code>${value}</code></div>`)
    .join("");
}

function updateModeHint() {
  const musicPattern = musicPatternSelect.value;
  if (musicPattern === "transient_sfx") {
    modeHint.textContent = "短提示音仍归纯语音类，不算背景音乐。";
    return;
  }
  if (musicPattern === "bgm" || musicPattern === "song") {
    modeHint.textContent = "这是持续音乐场景。后续处理方式由系统推断，不需要人工指定。";
    return;
  }
  modeHint.textContent = "这是纯语音场景。后续处理方式由系统推断，不需要人工指定。";
}

async function loadItems() {
  const response = await fetch("/api/items");
  const payload = await response.json();
  state.items = payload.items || [];
  state.index = 0;
  render();
}

function currentItem() {
  return state.items[state.index] || null;
}

function render() {
  const item = currentItem();
  if (!item) {
    counter.textContent = "没有待标注样本";
    sampleTitle.textContent = "";
    samplePath.textContent = "";
    sampleMeta.textContent = "";
    sampleSummary.innerHTML = "";
    outputSummary.textContent = "";
    player.removeAttribute("src");
    outputBlock.style.display = "none";
    return;
  }
  counter.textContent = `${state.index + 1} / ${state.items.length}`;
  sampleTitle.textContent = basename(item.path);
  samplePath.textContent = item.path;
  renderSummary(item);
  sampleMeta.textContent = JSON.stringify(item, null, 2);
  player.src = `/static-proxy?path=${encodeURIComponent(item.path)}`;
  if (item.output_path) {
    outputPlayer.src = `/static-proxy?path=${encodeURIComponent(item.output_path)}`;
    outputSummary.textContent = basename(item.output_path);
    outputBlock.style.display = "block";
  } else {
    outputSummary.textContent = "";
    outputBlock.style.display = "none";
  }
  document.getElementById("language").value = item.language || "mixed";
  speakerPatternSelect.value = item.speaker_pattern || "multi_speaker_other";
  document.getElementById("voice_age").value = item.voice_age || "unknown";
  musicPatternSelect.value = item.music_pattern || "no_music";
  updateModeHint();
}

async function saveCurrent() {
  const item = currentItem();
  if (!item) return;
  const labels = {
    language: document.getElementById("language").value,
    speaker_pattern: document.getElementById("speaker_pattern").value,
    voice_age: document.getElementById("voice_age").value,
    music_pattern: document.getElementById("music_pattern").value,
  };
  await fetch("/api/labels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_id: item.sample_id, labels, item }),
  });
  state.index += 1;
  render();
}

document.getElementById("saveBtn").addEventListener("click", saveCurrent);
musicPatternSelect.addEventListener("change", updateModeHint);
loadItems();
