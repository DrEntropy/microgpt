// microgpt educational UI - vanilla JS

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// State
let allSteps = [];       // full generation steps
let activeStepIdx = -1;  // which step is currently displayed
let stepMode = false;    // are we in step-through mode?
let stepTokenIds = [];   // token history for step mode

// Elements
const btnGenerate = $("#btn-generate");
const btnStep = $("#btn-step");
const btnNext = $("#btn-next");
const tempSlider = $("#temperature");
const tempVal = $("#temp-val");
const nameDisplay = $("#name-display");

// --- Helpers ---

function getTemp() {
    return parseFloat(tempSlider.value);
}

function softmax(logits, temp) {
    const scaled = logits.map((l) => l / temp);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map((v) => Math.exp(v - maxVal));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / total);
}

// Diverging color: blue (negative) -> white (0) -> red (positive)
function valueColor(v, maxAbs) {
    if (maxAbs === 0) return "rgb(255,255,255)";
    const t = Math.max(-1, Math.min(1, v / maxAbs));
    if (t >= 0) {
        const c = Math.round(255 * (1 - t));
        return `rgb(255,${c},${c})`;
    } else {
        const c = Math.round(255 * (1 + t));
        return `rgb(${c},${c},255)`;
    }
}

// --- Token strip ---

function buildTokenStrip() {
    const strip = $("#token-strip");
    strip.innerHTML = "";
    // BOS token
    const bos = document.createElement("span");
    bos.className = "token-chip";
    bos.textContent = "[START]";
    bos.dataset.id = META.BOS;
    strip.appendChild(bos);
    // Character tokens
    META.uchars.forEach((ch, i) => {
        const chip = document.createElement("span");
        chip.className = "token-chip";
        chip.textContent = ch;
        chip.dataset.id = i;
        strip.appendChild(chip);
    });
}

function highlightToken(tokenId) {
    $$(".token-chip").forEach((el) => {
        el.classList.toggle("active", parseInt(el.dataset.id) === tokenId);
    });
    const mapping = $("#token-mapping");
    if (tokenId === META.BOS) {
        mapping.textContent = "[START] \u2192 token " + tokenId;
    } else {
        mapping.textContent = `"${META.uchars[tokenId]}" \u2192 token ${tokenId}`;
    }
}

// --- Embedding visualization ---

function renderEmbedding(containerId, values) {
    const container = $(containerId);
    container.innerHTML = "";
    const maxAbs = Math.max(...values.map(Math.abs), 0.001);
    values.forEach((v) => {
        const cell = document.createElement("div");
        cell.className = "embed-cell";
        cell.style.backgroundColor = valueColor(v, maxAbs);
        const tip = document.createElement("span");
        tip.className = "tooltip";
        tip.textContent = v.toFixed(4);
        cell.appendChild(tip);
        container.appendChild(cell);
    });
}

// --- Combining card ---

function renderMlpGrid(values) {
    const container = $("#combine-mlp");
    container.innerHTML = "";
    const maxVal = Math.max(...values, 0.001);
    values.forEach((v) => {
        const cell = document.createElement("div");
        cell.className = "mlp-cell";
        // Green intensity: 0 = gray (dead neuron), max = bright teal
        const t = maxVal > 0 ? v / maxVal : 0;
        if (v === 0) {
            cell.style.backgroundColor = "#e2e8f0";
        } else {
            const r = Math.round(20 + (1 - t) * 200);
            const g = Math.round(184 - (1 - t) * 80);
            const b = Math.round(166 - (1 - t) * 60);
            cell.style.backgroundColor = `rgb(${r},${g},${b})`;
        }
        const tip = document.createElement("span");
        tip.className = "tooltip";
        tip.textContent = v === 0 ? "0 (dead)" : v.toFixed(4);
        cell.appendChild(tip);
        container.appendChild(cell);
    });
}

function renderCombining(inter) {
    renderEmbedding("#combine-post-attn", inter.layer0_post_attn);
    renderMlpGrid(inter.layer0_mlp_relu);
    renderEmbedding("#combine-final", inter.layer0_final_emb);
}

// --- Attention heatmaps ---

function initAttnHeads() {
    const container = $("#attn-heads");
    container.innerHTML = "";
    for (let h = 0; h < META.n_head; h++) {
        const div = document.createElement("div");
        div.className = "attn-head";
        const label = document.createElement("label");
        label.textContent = `Head ${h + 1}`;
        const canvas = document.createElement("canvas");
        canvas.id = `attn-canvas-${h}`;
        canvas.width = META.block_size;
        canvas.height = META.block_size;
        div.appendChild(label);
        div.appendChild(canvas);
        container.appendChild(div);
    }
}

// Accumulate attention weights across all steps
function renderAttention(steps, upToIdx) {
    const size = upToIdx + 1; // number of steps to show
    for (let h = 0; h < META.n_head; h++) {
        const canvas = $(`#attn-canvas-${h}`);
        canvas.width = Math.max(size, 1);
        canvas.height = Math.max(size, 1);
        const ctx = canvas.getContext("2d");
        const img = ctx.createImageData(size, size);

        for (let row = 0; row <= upToIdx; row++) {
            const aw = steps[row].intermediates.attn_weights[`layer0_head${h}`];
            for (let col = 0; col < aw.length; col++) {
                const intensity = Math.round(aw[col] * 255);
                const idx = (row * size + col) * 4;
                // Dark = high weight: invert so high weight = dark purple
                img.data[idx] = 139 - Math.round(intensity * 0.4);     // R
                img.data[idx + 1] = 92 - Math.round(intensity * 0.35); // G
                img.data[idx + 2] = 246;                                // B
                img.data[idx + 3] = Math.max(40, intensity);            // A
            }
        }
        ctx.putImageData(img, 0, 0);
    }
}

// --- Prediction bars ---

function renderPrediction(probs, sampledTokenId, logits) {
    const container = $("#pred-bars");
    container.innerHTML = "";

    // Build label-prob pairs
    const items = probs.map((p, i) => ({
        label: i === META.BOS ? "[END]" : META.uchars[i],
        prob: p,
        tokenId: i,
    }));
    items.sort((a, b) => b.prob - a.prob);

    // Show top 15 + sampled if not in top 15
    let shown = items.slice(0, 15);
    if (!shown.some((s) => s.tokenId === sampledTokenId)) {
        const sampled = items.find((s) => s.tokenId === sampledTokenId);
        if (sampled) shown.push(sampled);
    }

    const maxProb = Math.max(...shown.map((s) => s.prob), 0.001);

    shown.forEach((item) => {
        const row = document.createElement("div");
        row.className = "pred-row";

        const label = document.createElement("span");
        label.className = "pred-label";
        label.textContent = item.label;

        const barBg = document.createElement("div");
        barBg.className = "pred-bar-bg";
        const bar = document.createElement("div");
        bar.className = "pred-bar" + (item.tokenId === sampledTokenId ? " sampled" : "");
        bar.style.width = (item.prob / maxProb) * 100 + "%";
        barBg.appendChild(bar);

        const pct = document.createElement("span");
        pct.className = "pred-pct";
        pct.textContent = (item.prob * 100).toFixed(1) + "%";

        row.appendChild(label);
        row.appendChild(barBg);
        row.appendChild(pct);
        container.appendChild(row);
    });

    // Store logits for client-side temperature recomputation
    container.dataset.logits = JSON.stringify(logits);
    container.dataset.sampledId = sampledTokenId;
}

// --- Display a step ---

function showStep(idx) {
    if (idx < 0 || idx >= allSteps.length) return;
    activeStepIdx = idx;
    const step = allSteps[idx];

    // Highlight active char
    $$(".name-char").forEach((el, i) => {
        el.classList.toggle("active", i === idx);
    });

    // Tokenization
    highlightToken(step.input_token_id);

    // Embedding
    const inter = step.intermediates;
    renderEmbedding("#embed-tok", inter.tok_emb);
    renderEmbedding("#embed-pos", inter.pos_emb);
    renderEmbedding("#embed-combined", inter.combined_emb);

    // Attention (show cumulative up to this step)
    renderAttention(allSteps, idx);

    // Combining
    renderCombining(inter);

    // Prediction
    const temp = getTemp();
    const probs = softmax(inter.logits, temp);
    renderPrediction(probs, step.output_token_id, inter.logits);
}

// --- Name display ---

function renderName(steps) {
    nameDisplay.innerHTML = "";
    steps.forEach((step, i) => {
        const span = document.createElement("span");
        span.className = "name-char";
        span.textContent = step.output_char === "[END]" ? "\u23F9" : step.output_char;
        span.addEventListener("click", () => showStep(i));
        nameDisplay.appendChild(span);
    });
}

// --- Generate mode ---

async function doGenerate() {
    btnGenerate.disabled = true;
    btnStep.disabled = true;
    btnNext.disabled = true;
    stepMode = false;
    allSteps = [];
    activeStepIdx = -1;
    nameDisplay.innerHTML = '<span class="placeholder">Generating...</span>';

    const resp = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ temperature: getTemp() }),
    });
    const data = await resp.json();
    allSteps = data.steps;

    // Animate characters appearing
    nameDisplay.innerHTML = "";
    for (let i = 0; i < allSteps.length; i++) {
        const step = allSteps[i];
        const span = document.createElement("span");
        span.className = "name-char";
        span.textContent = step.output_char === "[END]" ? "\u23F9" : step.output_char;
        span.addEventListener("click", () => showStep(i));
        nameDisplay.appendChild(span);
        showStep(i);
        await new Promise((r) => setTimeout(r, 600));
    }

    btnGenerate.disabled = false;
    btnStep.disabled = false;
}

// --- Step-through mode ---

async function startStepMode() {
    stepMode = true;
    allSteps = [];
    activeStepIdx = -1;
    stepTokenIds = [META.BOS];
    nameDisplay.innerHTML = '<span class="placeholder">Click "Next Character" to generate one character at a time</span>';
    btnGenerate.disabled = true;
    btnStep.disabled = true;
    btnNext.disabled = false;
    initAttnHeads();
}

async function doNextStep() {
    btnNext.disabled = true;

    const resp = await fetch("/api/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token_ids: stepTokenIds, temperature: getTemp() }),
    });
    const step = await resp.json();
    allSteps.push(step);

    // Update name display
    renderName(allSteps);
    showStep(allSteps.length - 1);

    if (step.output_token_id === META.BOS) {
        // Name is done
        btnGenerate.disabled = false;
        btnStep.disabled = false;
        btnNext.disabled = true;
        stepMode = false;
    } else {
        stepTokenIds.push(step.output_token_id);
        btnNext.disabled = false;
    }
}

// --- Temperature slider ---

tempSlider.addEventListener("input", () => {
    tempVal.textContent = getTemp().toFixed(2);
    // If we have a step displayed, recompute predictions client-side
    if (activeStepIdx >= 0 && allSteps[activeStepIdx]) {
        const step = allSteps[activeStepIdx];
        const probs = softmax(step.intermediates.logits, getTemp());
        renderPrediction(probs, step.output_token_id, step.intermediates.logits);
    }
});

// --- Event listeners ---

btnGenerate.addEventListener("click", doGenerate);
btnStep.addEventListener("click", startStepMode);
btnNext.addEventListener("click", doNextStep);

// --- Init ---

buildTokenStrip();
initAttnHeads();
