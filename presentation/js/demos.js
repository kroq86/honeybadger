(function () {
  const state = {
    bundle: null,
    scenario: null,
    budgetIndex: 0
  };

  const FRIENDLY_SCENARIO_NAMES = {
    branch_on_equality: "Compare two values",
    countdown_steps: "Count down step by step",
    pair_sum_scan_3: "Find two numbers that make a sum",
    midpoint_search_4: "Move toward the middle",
    first_match_scan_4: "Find the first matching item"
  };

  const FRIENDLY_POLICY_NAMES = {
    none: "Bad order",
    heuristic: "Hand-made order",
    learned: "Learned order"
  };

  function normalizeBundle(payload) {
    if (Array.isArray(payload.scenarios)) return payload;
    return {
      dataset_path: payload.dataset_path,
      default_scenario_id: payload.scenario_id || `${payload.program_name}#${payload.record_index}`,
      scenarios: [payload]
    };
  }

  function bestPolicyFor(row) {
    const solved = row.policies.filter((policy) => policy.solved);
    if (!solved.length) return null;
    return solved.sort((a, b) => a.nodes_explored - b.nodes_explored)[0];
  }

  function friendlyScenarioLabel(scenario) {
    return FRIENDLY_SCENARIO_NAMES[scenario.program_name] || scenario.scenario_label || scenario.program_name;
  }

  function renderScenario() {
    const scenario = state.scenario;
    if (!scenario) return;

    const meta = document.getElementById("runtime-meta");
    const winner = document.getElementById("runtime-winner");
    const detail = document.getElementById("runtime-detail");
    const budgetRow = scenario.solve_with_budget_curve[state.budgetIndex];
    const best = bestPolicyFor(budgetRow);

    meta.textContent = `${friendlyScenarioLabel(scenario)} | effort levels: ${scenario.budgets.join(" / ")}`;
    winner.innerHTML = scenario.choose_next_step.winner
      ? `<strong>Checked winner in this scenario</strong><p>${scenario.choose_next_step.winner.instruction_text}</p><p>why it stands out here: it passed the checker</p>`
      : `<strong>No checked winner</strong><p>This example has no accepted first step under the current checker.</p>`;

    document.getElementById("runtime-policy-grid").innerHTML = budgetRow.policies
      .map((policy) => `
        <div class="policy-card ${best && best.policy === policy.policy ? "best" : ""}">
          <h3>${FRIENDLY_POLICY_NAMES[policy.policy] || policy.policy}</h3>
          <p>solved: <strong>${policy.solved ? "yes" : "no"}</strong></p>
          <p>effort used: <strong>${policy.nodes_explored}</strong></p>
          <p>limit reached: <strong>${policy.budget_exhausted ? "yes" : "no"}</strong></p>
        </div>
      `)
      .join("");

    detail.innerHTML = [
      `<p><strong>Current effort limit:</strong> ${budgetRow.node_budget}</p>`,
      `<p><strong>If it fails, the reason here is:</strong> ${scenario.failure_demo.category}</p>`,
      `<p><strong>Where the good option first appeared:</strong> ${scenario.failure_demo.first_hit_rank ?? "not found"}</p>`,
      `<p><strong>Learned path in this scenario:</strong> ${(scenario.compare_policies.policies.find((item) => item.policy === "learned")?.successful_path || []).join(" -> ") || "none yet"}</p>`
    ].join("");

    renderChart(scenario);
  }

  function renderChart(scenario) {
    const target = document.getElementById("runtime-budget-chart");
    if (!target || typeof Plotly === "undefined") return;

    const budgets = scenario.solve_with_budget_curve.map((row) => row.node_budget);
    const policyNames = ["none", "heuristic", "learned"];
    const traces = policyNames
      .map((policy) => {
        const y = scenario.solve_with_budget_curve.map((row) => {
          const item = row.policies.find((candidate) => candidate.policy === policy);
          return item ? item.nodes_explored : null;
        });
        if (y.every((item) => item === null)) return null;
        return {
          x: budgets,
          y,
          type: "scatter",
          mode: "lines+markers",
          name: FRIENDLY_POLICY_NAMES[policy] || policy,
          line: {
            width: 3
          }
        };
      })
      .filter(Boolean);

    const selectedBudget = scenario.solve_with_budget_curve[state.budgetIndex].node_budget;
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(17,25,34,0.7)",
      font: { family: "Space Grotesk, sans-serif", color: "#edf3f8", size: 12 },
      margin: { l: 48, r: 18, t: 20, b: 50 },
      xaxis: { title: "effort limit", gridcolor: "rgba(154,181,204,0.10)" },
      yaxis: { title: "effort used", gridcolor: "rgba(154,181,204,0.10)" },
      shapes: [
        {
          type: "line",
          x0: selectedBudget,
          x1: selectedBudget,
          y0: 0,
          y1: 1,
          xref: "x",
          yref: "paper",
          line: { color: "#f7c66a", width: 2, dash: "dot" }
        }
      ],
      legend: { orientation: "h", x: 0, y: 1.14 }
    };

    Plotly.react(target, traces, layout, {
      responsive: true,
      displayModeBar: false
    });
  }

  function setScenario(id) {
    const bundle = state.bundle;
    state.scenario = bundle.scenarios.find((item) => (item.scenario_id || `${item.program_name}#${item.record_index}`) === id) || bundle.scenarios[0];
    state.budgetIndex = 0;
    const slider = document.getElementById("runtime-budget");
    slider.max = String(Math.max(state.scenario.solve_with_budget_curve.length - 1, 0));
    slider.value = "0";
    renderScenario();
  }

  async function bootRuntimeDemo() {
    const select = document.getElementById("runtime-scenario");
    const slider = document.getElementById("runtime-budget");
    if (!select || !slider) return;

    const response = await fetch("./assets/demo-runtime-payload.json");
    const envelope = await response.json();
    state.bundle = normalizeBundle(envelope.payload);

    select.innerHTML = state.bundle.scenarios
      .map((scenario) => {
        const id = scenario.scenario_id || `${scenario.program_name}#${scenario.record_index}`;
        const label = friendlyScenarioLabel(scenario);
        return `<option value="${id}">${label}</option>`;
      })
      .join("");

    select.addEventListener("change", () => setScenario(select.value));
    slider.addEventListener("input", () => {
      state.budgetIndex = Number(slider.value);
      renderScenario();
    });

    setScenario(state.bundle.default_scenario_id || select.value);
    select.value = (state.scenario.scenario_id || `${state.scenario.program_name}#${state.scenario.record_index}`);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootRuntimeDemo);
  } else {
    bootRuntimeDemo();
  }
})();
