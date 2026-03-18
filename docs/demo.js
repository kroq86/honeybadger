let payloadBundle = null;
let activeScenario = null;
let selectedBudgetIndex = 0;
let selectedCandidateRank = 1;
let selectedPolicyName = "learned";

async function loadDemo() {
  const response = await fetch("./demo-runtime-payload.json");
  const envelope = await response.json();
  payloadBundle = normalizeBundle(envelope.payload);
  setupScenarioSelector(payloadBundle);
  setActiveScenario(payloadBundle.default_scenario_id || payloadBundle.scenarios[0]?.scenario_id);
}

function normalizeBundle(payload) {
  if (Array.isArray(payload.scenarios)) {
    return payload;
  }
  const singleScenario = {
    ...payload,
    scenario_id: payload.scenario_id || `${payload.program_name}#${payload.record_index}`,
    scenario_label: payload.scenario_label || `${payload.program_name} #${payload.record_index}`,
  };
  return {
    dataset_path: payload.dataset_path,
    record_indices: [payload.record_index],
    default_scenario_id: singleScenario.scenario_id,
    scenario_count: 1,
    scenarios: [singleScenario],
  };
}

function setupScenarioSelector(bundle) {
  const select = document.getElementById("scenario-select");
  select.innerHTML = bundle.scenarios
    .map(
      (scenario) =>
        `<option value="${escapeHtml(scenario.scenario_id)}">${escapeHtml(scenario.scenario_label)}</option>`
    )
    .join("");
  select.addEventListener("change", () => {
    setActiveScenario(select.value);
  });
}

function setActiveScenario(scenarioId) {
  const scenario = payloadBundle.scenarios.find((item) => item.scenario_id === scenarioId) || payloadBundle.scenarios[0];
  if (!scenario) {
    return;
  }
  activeScenario = scenario;
  selectedBudgetIndex = 0;
  selectedCandidateRank = 1;
  selectedPolicyName = scenario.compare_policies.policies.some((item) => item.policy === "learned") ? "learned" : scenario.compare_policies.policies[0].policy;

  const select = document.getElementById("scenario-select");
  select.value = scenario.scenario_id;
  document.getElementById("demo-meta").textContent =
    `${scenario.program_name} | record=${scenario.record_index} | verifier=${scenario.verifier_mode} | budgets=${scenario.budgets.join("/")}`;

  renderNextStep(scenario.choose_next_step);
  renderFailure(scenario.failure_demo);
  renderPolicyCompare(scenario.compare_policies.policies);
  renderBudgetCurve(scenario.solve_with_budget_curve);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderNextStep(section) {
  const winner = section.winner;
  const list = document.getElementById("next-step-list");

  document.getElementById("next-step-winner").innerHTML = winner
    ? `<strong>${escapeHtml(winner.instruction_text)}</strong><span class="pill success">verified winner</span><span class="pill">${escapeHtml(winner.source)}</span>`
    : `<strong>No verified winner</strong>`;

  list.innerHTML = section.top_candidates
    .map(
      (candidate) => `
        <button class="candidate-row ${candidate.rank === selectedCandidateRank ? "active" : ""}" data-rank="${candidate.rank}">
          <div>
            <strong>#${candidate.rank} ${escapeHtml(candidate.instruction_text)}</strong>
            <div class="candidate-meta">${escapeHtml(candidate.source)}</div>
          </div>
          <div class="candidate-status">
            <span class="pill ${candidate.verified ? "success" : "danger"}">${candidate.verified ? "verified" : "rejected"}</span>
          </div>
        </button>
      `
    )
    .join("");

  list.querySelectorAll(".candidate-row").forEach((node) => {
    node.addEventListener("click", () => {
      selectedCandidateRank = Number(node.dataset.rank);
      renderNextStep(section);
    });
  });

  const selected = section.top_candidates.find((candidate) => candidate.rank === selectedCandidateRank) || section.top_candidates[0];
  if (selected) {
    selectedCandidateRank = selected.rank;
  }
  document.getElementById("candidate-detail").innerHTML = selected
    ? `
      <div class="detail-title">Candidate detail</div>
      <div class="detail-line"><strong>instruction</strong> ${escapeHtml(selected.instruction_text)}</div>
      <div class="detail-line"><strong>source</strong> ${escapeHtml(selected.source)}</div>
      <div class="detail-line"><strong>verified</strong> ${selected.verified}</div>
      <div class="detail-line"><strong>matches first target state</strong> ${selected.matches_first_target_state}</div>
      <div class="detail-line"><strong>notes</strong> ${(selected.notes || []).length ? selected.notes.map(escapeHtml).join(", ") : "none"}</div>
    `
    : `<div class="detail-title">Candidate detail</div><div class="detail-line">No candidate selected.</div>`;
}

function renderFailure(section) {
  const details = section.details;
  const attempts = details.attempts || [];
  document.getElementById("failure-view").innerHTML = `
    <div class="failure-card">
      <div class="failure-title">${escapeHtml(section.category)}</div>
      <div class="failure-copy">nodes=${details.nodes_explored} | budget_exhausted=${details.budget_exhausted}</div>
      <div class="failure-copy">first_hit_rank=${section.first_hit_rank ?? "n/a"}</div>
      <div class="attempt-list">
        ${attempts
          .map(
            (attempt, index) => `
              <div class="attempt-row">
                <div class="attempt-head">attempt ${index + 1}: ${escapeHtml(attempt.instruction_text)}</div>
                <div class="attempt-meta">${escapeHtml(attempt.source)} | expected=${escapeHtml(attempt.expected_instruction || "n/a")}</div>
                <div class="attempt-meta">${(attempt.notes || []).length ? attempt.notes.map(escapeHtml).join(", ") : "no notes"}</div>
              </div>
            `
          )
          .join("")}
      </div>
    </div>
  `;
}

function renderPolicyCompare(policies) {
  const container = document.getElementById("policy-compare");
  container.innerHTML = policies
    .map(
      (policy) => `
        <button class="policy-card ${policy.policy === selectedPolicyName ? "active" : ""}" data-policy="${escapeHtml(policy.policy)}">
          <div class="policy-name">${escapeHtml(policy.policy)}</div>
          <div class="policy-metric">solved: <strong>${policy.solved}</strong></div>
          <div class="policy-metric">nodes: <strong>${policy.nodes_explored}</strong></div>
          <div class="policy-metric">budget exhausted: <strong>${policy.budget_exhausted}</strong></div>
          <div class="policy-path">${(policy.successful_path || []).map(escapeHtml).join(" -> ") || "no path"}</div>
        </button>
      `
    )
    .join("");

  container.querySelectorAll(".policy-card").forEach((node) => {
    node.addEventListener("click", () => {
      selectedPolicyName = node.dataset.policy;
      renderPolicyCompare(policies);
    });
  });

  const selected = policies.find((policy) => policy.policy === selectedPolicyName) || policies[0];
  if (selected) {
    selectedPolicyName = selected.policy;
  }
  document.getElementById("policy-detail").innerHTML = selected
    ? `
      <div class="detail-title">Policy trace</div>
      <div class="detail-line"><strong>policy</strong> ${escapeHtml(selected.policy)}</div>
      <div class="detail-line"><strong>path</strong> ${(selected.successful_path || []).map(escapeHtml).join(" -> ") || "none"}</div>
      <div class="attempt-list">
        ${(selected.attempts || [])
          .map(
            (attempt, index) => `
              <div class="attempt-row">
                <div class="attempt-head">attempt ${index + 1}: ${escapeHtml(attempt.instruction_text)}</div>
                <div class="attempt-meta">${escapeHtml(attempt.source)} | expected=${escapeHtml(attempt.expected_instruction || "n/a")}</div>
                <div class="attempt-meta">${(attempt.notes || []).length ? attempt.notes.map(escapeHtml).join(", ") : "no notes"}</div>
              </div>
            `
          )
          .join("")}
      </div>
    `
    : `<div class="detail-title">Policy trace</div><div class="detail-line">No policy selected.</div>`;
}

function renderBudgetCurve(rows) {
  const controls = document.getElementById("budget-controls");
  const view = document.getElementById("budget-view");
  const slider = document.getElementById("budget-slider");

  slider.max = String(Math.max(rows.length - 1, 0));
  slider.value = String(selectedBudgetIndex);

  controls.innerHTML = rows
    .map(
      (row, index) => `
        <button class="budget-button ${index === selectedBudgetIndex ? "active" : ""}" data-index="${index}">
          budget ${row.node_budget}
        </button>
      `
    )
    .join("");

  function paintBudget(index) {
    selectedBudgetIndex = index;
    slider.value = String(index);
    controls.querySelectorAll(".budget-button").forEach((node, nodeIndex) => {
      node.classList.toggle("active", nodeIndex === index);
    });
    const row = rows[index];
    view.innerHTML = `
      <div class="budget-summary">budget ${row.node_budget}</div>
      <div class="policy-grid">
        ${row.policies
          .map(
            (policy) => `
              <div class="policy-card compact">
                <div class="policy-name">${escapeHtml(policy.policy)}</div>
                <div class="policy-metric">solved: <strong>${policy.solved}</strong></div>
                <div class="policy-metric">nodes: <strong>${policy.nodes_explored}</strong></div>
                <div class="policy-metric">budget exhausted: <strong>${policy.budget_exhausted}</strong></div>
                <div class="policy-path">${(policy.successful_path || []).map(escapeHtml).join(" -> ") || "no path"}</div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  controls.querySelectorAll(".budget-button").forEach((button) => {
    button.addEventListener("click", () => {
      paintBudget(Number(button.dataset.index));
    });
  });

  slider.addEventListener("input", () => {
    paintBudget(Number(slider.value));
  });

  paintBudget(selectedBudgetIndex);
}

loadDemo().catch((error) => {
  const target = document.getElementById("demo-meta");
  target.textContent = `demo load failed: ${error.message}`;
});
