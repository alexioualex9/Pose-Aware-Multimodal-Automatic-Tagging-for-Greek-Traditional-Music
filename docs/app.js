const select = document.querySelector("#model-select");
const selectedResult = document.querySelector("#selected-result");
const comparisonChart = document.querySelector("#comparison-chart");
const datasetDescription = document.querySelector("#dataset-description");
const metricLabel = document.querySelector("#metric-label");

let results;

function formatScore(value) {
  return Number(value).toFixed(3);
}

function renderSelectedModel(model) {
  const audioBaseline = results.models.find((item) => item.id === "audio");
  const difference = model.roc_auc - audioBaseline.roc_auc;
  const differenceText = difference >= 0
    ? `+${formatScore(difference)} compared with Audio (AST)`
    : `${formatScore(difference)} compared with Audio (AST)`;

  selectedResult.innerHTML = `
    <div class="result-layout">
      <div>
        <h3 class="result-title">${model.name}</h3>
        <p class="result-description">${model.description}</p>
        <div class="chips" aria-label="Modalities used">
          ${model.modalities.map((modality) => `<span class="chip">${modality}</span>`).join("")}
        </div>
      </div>
      <div class="score-box">
        <span class="score-label">${results.metric}</span>
        <strong class="score">${formatScore(model.roc_auc)}</strong>
        <p class="gain">${differenceText}</p>
      </div>
    </div>
  `;
}

function renderComparison(selectedId) {
  const minimum = 0.50;
  const maximum = 0.90;

  comparisonChart.innerHTML = results.models.map((model) => {
    const width = Math.max(0, Math.min(100, ((model.roc_auc - minimum) / (maximum - minimum)) * 100));
    const selectedClass = model.id === selectedId ? "is-selected" : "";

    return `
      <div class="chart-row ${selectedClass}">
        <span class="chart-name">${model.name}</span>
        <div class="chart-track" aria-hidden="true">
          <div class="chart-bar" style="width: ${width}%"></div>
        </div>
        <span class="chart-value">${formatScore(model.roc_auc)}</span>
      </div>
    `;
  }).join("");
}

function updateView() {
  const selectedModel = results.models.find((model) => model.id === select.value);
  renderSelectedModel(selectedModel);
  renderComparison(selectedModel.id);
}

async function initialise() {
  try {
    const response = await fetch("data/results.json");
    if (!response.ok) throw new Error("Could not load results.json");

    results = await response.json();
    document.title = results.title;
    datasetDescription.textContent = results.dataset;
    metricLabel.textContent = `Metric: ${results.metric}`;

    select.innerHTML = results.models.map((model) => (
      `<option value="${model.id}">${model.name}</option>`
    )).join("");

    select.disabled = false;
    select.addEventListener("change", updateView);
    updateView();
  } catch (error) {
    selectedResult.innerHTML = `
      <p><strong>The demo data could not be loaded.</strong></p>
      <p>Run the site through a local web server or GitHub Pages, then check that <code>data/results.json</code> exists.</p>
    `;
    console.error(error);
  }
}

initialise();
