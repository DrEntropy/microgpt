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

function renderHeadOutputs(inter) {
    const container = $("#combine-heads");
    container.innerHTML = "";
    const values = inter.layer0_head_outputs;
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

function renderProjection(inter) {
    // Compact final embedding
    renderEmbedding("#proj-embed", inter.layer0_final_emb);

    // Logit cells
    const container = $("#proj-logits");
    container.innerHTML = "";
    const logits = inter.logits;
    const maxAbs = Math.max(...logits.map(Math.abs), 0.001);

    // Find top 5 indices
    const indexed = logits.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    const top5 = new Set(indexed.slice(0, 5).map((x) => x.i));

    logits.forEach((v, i) => {
        const wrapper = document.createElement("div");
        wrapper.style.display = "flex";
        wrapper.style.flexDirection = "column";
        wrapper.style.alignItems = "center";

        const cell = document.createElement("div");
        cell.className = "logit-cell";
        cell.style.backgroundColor = valueColor(v, maxAbs);
        const charLabel = i === META.BOS ? "[END]" : META.uchars[i];
        const tip = document.createElement("span");
        tip.className = "tooltip";
        tip.textContent = `${charLabel}: ${v.toFixed(2)}`;
        cell.appendChild(tip);
        wrapper.appendChild(cell);

        if (top5.has(i)) {
            const lbl = document.createElement("div");
            lbl.className = "logit-label";
            lbl.textContent = charLabel;
            wrapper.appendChild(lbl);
        }

        container.appendChild(wrapper);
    });
}

function renderCombining(inter) {
    renderHeadOutputs(inter);
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
        const gridWrap = document.createElement("div");
        gridWrap.className = "attn-grid-wrap";
        const rowLabels = document.createElement("div");
        rowLabels.className = "attn-row-labels";
        rowLabels.id = `attn-row-labels-${h}`;
        const colWrap = document.createElement("div");
        colWrap.className = "attn-col-wrap";
        const canvas = document.createElement("canvas");
        canvas.id = `attn-canvas-${h}`;
        canvas.width = META.block_size;
        canvas.height = META.block_size;
        const colLabels = document.createElement("div");
        colLabels.className = "attn-col-labels";
        colLabels.id = `attn-col-labels-${h}`;
        colWrap.appendChild(canvas);
        colWrap.appendChild(colLabels);
        gridWrap.appendChild(rowLabels);
        gridWrap.appendChild(colWrap);
        div.appendChild(label);
        div.appendChild(gridWrap);
        container.appendChild(div);
    }
}

// Accumulate attention weights across all steps
function renderAttention(steps, upToIdx) {
    const size = upToIdx + 1; // number of steps to show
    // Collect token labels for each position
    const chars = [];
    for (let i = 0; i <= upToIdx; i++) {
        const ch = steps[i].input_char;
        chars.push(ch === "[START]" ? "\u25B6" : ch);
    }

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

        // Row labels (left side)
        const rowLabels = $(`#attn-row-labels-${h}`);
        rowLabels.innerHTML = "";
        chars.forEach((ch) => {
            const span = document.createElement("span");
            span.textContent = ch;
            rowLabels.appendChild(span);
        });

        // Column labels (bottom)
        const colLabels = $(`#attn-col-labels-${h}`);
        colLabels.innerHTML = "";
        chars.forEach((ch) => {
            const span = document.createElement("span");
            span.textContent = ch;
            colLabels.appendChild(span);
        });
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

    // Projection
    renderProjection(inter);

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

// --- Tab switching ---

let weightsData = null;
let weightsLoaded = false;

$$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.tab;
        $$(".tab-panel").forEach((p) => {
            p.style.display = p.id === target ? "" : "none";
        });
        if (target === "tab-weights" && !weightsLoaded) {
            loadWeights();
        }
    });
});

// --- Weights tab ---

const WEIGHT_DEFS = [
    { key: "wte", title: "Token Embeddings (wte)", desc: "Each row is the learned 16-dim vector for one character", color: "blue", rowLabels: true },
    { key: "wpe", title: "Position Embeddings (wpe)", desc: "Each row encodes one of the 16 possible positions", color: "blue", rowLabels: "pos" },
    { key: "layer0.attn_wq", title: "Query Projection (Wq)", desc: "Projects input into query vectors for all 4 heads", color: "purple" },
    { key: "layer0.attn_wk", title: "Key Projection (Wk)", desc: "Projects input into key vectors for all 4 heads", color: "purple" },
    { key: "layer0.attn_wv", title: "Value Projection (Wv)", desc: "Projects input into value vectors for all 4 heads", color: "purple" },
    { key: "layer0.attn_wo", title: "Output Projection (Wo)", desc: "Merges head outputs back into embedding space", color: "purple" },
    { key: "layer0.mlp_fc1", title: "MLP Up (fc1)", desc: "Expands 16-dim embedding to 64-dim hidden layer", color: "teal" },
    { key: "layer0.mlp_fc2", title: "MLP Down (fc2)", desc: "Projects 64-dim hidden back to 16-dim embedding", color: "teal" },
    { key: "lm_head", title: "Language Model Head (lm_head)", desc: "Maps final embedding to 27 logits (one per character)", color: "rose", rowLabels: true },
];

async function loadWeights() {
    const resp = await fetch("/api/weights");
    weightsData = await resp.json();
    weightsLoaded = true;
    $("#weights-loading").style.display = "none";
    renderAllWeights();
}

function renderAllWeights() {
    const container = $("#weights-cards");
    container.innerHTML = "";

    WEIGHT_DEFS.forEach((def) => {
        const matrix = weightsData[def.key];
        if (!matrix) return;

        const rows = matrix.length;
        const cols = matrix[0].length;

        // Card structure matching existing pattern
        const card = document.createElement("div");
        card.className = "card";

        const header = document.createElement("div");
        header.className = "card-header " + def.color;
        header.innerHTML = `<h2>${def.title}</h2><p>${def.desc}</p>`;
        card.appendChild(header);

        const body = document.createElement("div");
        body.className = "card-body";

        const shape = document.createElement("div");
        shape.className = "weight-shape";
        shape.textContent = `Shape: ${rows} \u00D7 ${cols}`;
        body.appendChild(shape);

        const wrap = document.createElement("div");
        wrap.className = "weight-canvas-wrap";

        // Row labels for wte, lm_head, wpe
        if (def.rowLabels) {
            const labelsDiv = document.createElement("div");
            labelsDiv.className = "weight-row-labels";
            for (let r = 0; r < rows; r++) {
                const span = document.createElement("span");
                if (def.rowLabels === "pos") {
                    span.textContent = r;
                } else {
                    span.textContent = r === META.BOS ? "\u25B6" : META.uchars[r];
                }
                labelsDiv.appendChild(span);
            }
            wrap.appendChild(labelsDiv);
        }

        const inner = document.createElement("div");
        inner.className = "weight-canvas-inner";

        const canvas = document.createElement("canvas");
        canvas.width = cols;
        canvas.height = rows;
        inner.appendChild(canvas);
        wrap.appendChild(inner);

        // Tooltip
        const tooltip = document.createElement("div");
        tooltip.className = "weight-tooltip";
        wrap.appendChild(tooltip);

        body.appendChild(wrap);
        card.appendChild(body);
        container.appendChild(card);

        drawWeightCanvas(canvas, matrix, rows, cols);

        // Mouse tooltip
        canvas.addEventListener("mousemove", (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const col = Math.floor((x / rect.width) * cols);
            const row = Math.floor((y / rect.height) * rows);
            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                const val = matrix[row][col];
                let rowLabel = `row ${row}`;
                if (def.rowLabels === "pos") {
                    rowLabel = `pos ${row}`;
                } else if (def.rowLabels === true) {
                    rowLabel = row === META.BOS ? "[BOS]" : META.uchars[row];
                }
                tooltip.textContent = `[${rowLabel}, col ${col}] = ${val.toFixed(4)}`;
                tooltip.style.display = "block";
                tooltip.style.left = (x + 12) + "px";
                tooltip.style.top = (y - 24) + "px";
            }
        });
        canvas.addEventListener("mouseleave", () => {
            tooltip.style.display = "none";
        });
    });
}

function drawWeightCanvas(canvas, matrix, rows, cols) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(cols, rows);

    // Find max absolute value for this matrix
    let maxAbs = 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const a = Math.abs(matrix[r][c]);
            if (a > maxAbs) maxAbs = a;
        }
    }
    if (maxAbs === 0) maxAbs = 1;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const v = matrix[r][c];
            const t = Math.max(-1, Math.min(1, v / maxAbs));
            const idx = (r * cols + c) * 4;
            if (t >= 0) {
                img.data[idx] = 255;
                img.data[idx + 1] = Math.round(255 * (1 - t));
                img.data[idx + 2] = Math.round(255 * (1 - t));
            } else {
                img.data[idx] = Math.round(255 * (1 + t));
                img.data[idx + 1] = Math.round(255 * (1 + t));
                img.data[idx + 2] = 255;
            }
            img.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

// --- Init ---

buildTokenStrip();
initAttnHeads();
