/**
 * Quantitative Trading Web Demonstration Application
 * Course: UE24CS352A - Machine Learning Mini-Project
 * Team: Rottela Haritej (PES1UG24AM231) & Pradyun P Urs (PES1UG24AM195)
 */

document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------------------
  // 1. Asset Configuration for Day-t Shock Inputs (Lag 0)
  // -------------------------------------------------------------------------
  const ASSETS = [
    { id: "CON_DE", name: "Continental AG", ticker: "CON.DE", role: "Tires & Braking" },
    { id: "IFX_DE", name: "Infineon Tech", ticker: "IFX.DE", role: "Automotive Chips" },
    { id: "SHA0_DE", name: "Schaeffler AG", ticker: "SHA0.DE", role: "Transmission" },
    { id: "FR_PA", name: "Valeo SE", ticker: "FR.PA", role: "Electronics/Sensors" },
    { id: "BAS_DE", name: "BASF SE", ticker: "BAS.DE", role: "Coatings & Plastics" },
    { id: "ML_PA", name: "Michelin", ticker: "ML.PA", role: "OEM Tires" },
    { id: "ZIL2_DE", name: "ElringKlinger", ticker: "ZIL2.DE", role: "Gaskets & Battery" },
    { id: "GDAXI", name: "DAX 40 Index", ticker: "^GDAXI", role: "German Benchmark" },
    { id: "EURUSD_X", name: "EUR/USD FX", ticker: "EURUSD=X", role: "Export FX (Lag>=1)" },
    { id: "VOW3_DE", name: "VW (Current Day)", ticker: "VOW3.DE", role: "Target Momentum" }
  ];

  const PRESETS = {
    baseline: {
      CON_DE: 0.0, IFX_DE: 0.0, SHA0_DE: 0.0, FR_PA: 0.0, BAS_DE: 0.0,
      ML_PA: 0.0, ZIL2_DE: 0.0, GDAXI: 0.0, EURUSD_X: 0.0, VOW3_DE: 0.0
    },
    semi_crunch: {
      CON_DE: -1.8, IFX_DE: -3.8, SHA0_DE: -1.2, FR_PA: -2.0, BAS_DE: -0.5,
      ML_PA: -0.2, ZIL2_DE: -1.5, GDAXI: -0.9, EURUSD_X: 0.3, VOW3_DE: -1.4
    },
    sector_rally: {
      CON_DE: 2.8, IFX_DE: 3.4, SHA0_DE: 2.1, FR_PA: 2.9, BAS_DE: 1.2,
      ML_PA: 1.8, ZIL2_DE: 1.5, GDAXI: 1.6, EURUSD_X: -0.5, VOW3_DE: 2.2
    },
    chemical_shock: {
      CON_DE: -0.6, IFX_DE: -0.2, SHA0_DE: -0.8, FR_PA: -1.1, BAS_DE: -3.5,
      ML_PA: -1.5, ZIL2_DE: -0.4, GDAXI: -0.8, EURUSD_X: -1.2, VOW3_DE: -0.9
    },
    fx_volatility: {
      CON_DE: 0.4, IFX_DE: 0.2, SHA0_DE: 0.1, FR_PA: 0.3, BAS_DE: -0.2,
      ML_PA: 0.5, ZIL2_DE: 0.0, GDAXI: -0.4, EURUSD_X: 2.6, VOW3_DE: -0.8
    }
  };

  // State
  const sliderValues = {};
  ASSETS.forEach(a => sliderValues[a.id] = 0.0);

  // DOM Elements
  const slidersContainer = document.getElementById("slidersContainer");
  const signalBanner = document.getElementById("signalBanner");
  const signalType = document.getElementById("signalType");
  const signalTargetVal = document.getElementById("signalTargetVal");
  const signalSubText = document.getElementById("signalSubText");
  const signalPosition = document.getElementById("signalPosition");
  const signalConfidence = document.getElementById("signalConfidence");
  const modelLabelVal = document.getElementById("modelLabelVal");
  const modelStatusBadge = document.getElementById("modelStatusBadge");
  const attributionList = document.getElementById("attributionList");
  const thresholdSelect = document.getElementById("thresholdSelect");
  const scenarioBtns = document.querySelectorAll(".scenario-btn");

  // Build Sliders dynamically
  function buildSliders() {
    if (!slidersContainer) return;
    slidersContainer.innerHTML = "";
    ASSETS.forEach(asset => {
      const item = document.createElement("div");
      item.className = "slider-item";
      item.innerHTML = `
        <div class="slider-header">
          <div>
            <span class="asset-name">${asset.name}</span>
            <span class="asset-ticker">${asset.ticker} &bull; ${asset.role}</span>
          </div>
          <span class="slider-val" id="val_${asset.id}">0.0%</span>
        </div>
        <input type="range" class="custom-range" id="range_${asset.id}" min="-5.0" max="5.0" step="0.1" value="0.0">
      `;
      slidersContainer.appendChild(item);

      const input = item.querySelector("input");
      const valDisplay = item.querySelector(`#val_${asset.id}`);

      input.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        sliderValues[asset.id] = val;
        valDisplay.textContent = (val > 0 ? "+" : "") + val.toFixed(1) + "%";
        valDisplay.style.color = val > 0 ? "var(--green)" : val < 0 ? "var(--red)" : "var(--text-dim)";
        triggerModelInference();
      });
    });
  }

  function applyPreset(presetKey) {
    const preset = PRESETS[presetKey];
    if (!preset) return;
    ASSETS.forEach(asset => {
      const val = preset[asset.id] !== undefined ? preset[asset.id] : 0.0;
      sliderValues[asset.id] = val;
      const input = document.getElementById(`range_${asset.id}`);
      const valDisplay = document.getElementById(`val_${asset.id}`);
      if (input) input.value = val;
      if (valDisplay) {
        valDisplay.textContent = (val > 0 ? "+" : "") + val.toFixed(1) + "%";
        valDisplay.style.color = val > 0 ? "var(--green)" : val < 0 ? "var(--red)" : "var(--text-dim)";
      }
    });
    triggerModelInference();
  }

  scenarioBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      scenarioBtns.forEach(b => b.classList.toggle("active", b === btn));
      applyPreset(btn.dataset.preset);
    });
  });

  const modelSelect = document.getElementById("modelSelect");
  if (modelSelect) {
    modelSelect.addEventListener("change", () => triggerModelInference());
  }

  if (thresholdSelect) {
    thresholdSelect.addEventListener("change", () => triggerModelInference());
  }

  // -------------------------------------------------------------------------
  // 2. Real ML Model Inference (/api/predict)
  // -------------------------------------------------------------------------
  let debounceTimer = null;

  function triggerModelInference() {
    clearTimeout(debounceTimer);

    debounceTimer = setTimeout(() => {
      const thresholdVal = parseFloat(thresholdSelect ? thresholdSelect.value : 0.0025);
      const modelVal = modelSelect ? modelSelect.value : "linear_reg";

      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          returns: sliderValues,
          threshold: thresholdVal,
          model: modelVal
        })
      })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data && data.success) {
          renderRealModelInference(data);
        } else {
          throw new Error(data.error || "Prediction failed");
        }
      })
      .catch(err => {
        console.warn("Backend inference notice:", err);
        handleInferenceError(err);
      });
    }, 20);
  }

  function renderRealModelInference(data) {
    // 1. Status Badge
    if (modelStatusBadge) {
      modelStatusBadge.innerHTML = `<span class="status-dot"></span><span>Python Pipeline Active (${data.model})</span>`;
    }

    // 2. Prediction Value & Signal Banner
    const predPct = data.predicted_return_pct;
    const isPos = predPct > 0;
    signalTargetVal.textContent = (isPos ? "+" : "") + predPct.toFixed(3) + "%";

    const sig = data.signal.toUpperCase();
    signalBanner.className = "signal-banner " + sig.toLowerCase();

    if (sig === "LONG") {
      signalType.textContent = "LONG SIGNAL";
      signalPosition.textContent = "+1.0 (LONG)";
      signalPosition.style.color = "var(--green)";
    } else if (sig === "SHORT") {
      signalType.textContent = "SHORT SIGNAL";
      signalPosition.textContent = "-1.0 (SHORT)";
      signalPosition.style.color = "var(--red)";
    } else {
      signalType.textContent = "FLAT / CASH";
      signalPosition.textContent = "0.0 (CASH)";
      signalPosition.style.color = "var(--amber)";
    }

    // 3. Conviction Ratio (|Forecast| / Threshold)
    const ratio = data.conviction_ratio !== undefined ? data.conviction_ratio : (Math.abs(data.predicted_return) / Math.max(data.threshold, 0.0001));
    if (sig === "FLAT") {
      signalConfidence.textContent = `${ratio.toFixed(2)}x (Below c)`;
      signalConfidence.style.color = "var(--amber)";
    } else {
      signalConfidence.textContent = `${ratio.toFixed(2)}x (Active)`;
      signalConfidence.style.color = "var(--green)";
    }

    if (signalSubText) {
      if (data.model_key === "elastic_net_sparse") {
        signalSubText.textContent = `Sparse Elastic Net: L1 penalty pruned all weights to 0 (Constant Cash). Select 'Linear Regression' to see active factor elasticity.`;
      } else if (data.model_key === "decision_tree") {
        signalSubText.textContent = `Decision Tree (h=5d): Splits primarily on DAX & historical lags. Select 'Linear Regression' for direct supplier sensitivity.`;
      } else {
        signalSubText.textContent = `Forecast: ${predPct > 0 ? '+' : ''}${predPct.toFixed(3)}% | Threshold c: ${(data.threshold * 100).toFixed(2)}% | Date: ${data.feature_date}`;
      }
    }

    if (modelLabelVal) {
      modelLabelVal.textContent = `${data.model} (h=${data.horizon_trading_days}d)`;
    }

    // 4. Attribution list from model beta_k * shock_k
    renderAttributions(data.attributions || []);
  }

  function handleInferenceError(err) {
    if (modelStatusBadge) {
      modelStatusBadge.innerHTML = `<span class="status-dot" style="background: var(--red);"></span><span>Server Offline (Start server.py)</span>`;
    }
    if (signalSubText) {
      signalSubText.textContent = `Offline: Please launch 'python web/server.py' for live model execution.`;
    }
  }

  function renderAttributions(topAtts) {
    if (!attributionList) return;
    attributionList.innerHTML = "";
    if (!topAtts || topAtts.length === 0) {
      attributionList.innerHTML = `<div style="font-size: 0.8rem; color: var(--text-dim); padding: 0.5rem 0;">No active non-zero supplier shocks injected for lag 0.</div>`;
      return;
    }

    const maxVal = Math.max(...topAtts.map(a => a.abs_impact), 0.0001);

    topAtts.forEach(att => {
      const row = document.createElement("div");
      row.className = "attribution-bar-row";
      const isPos = att.impact_pct >= 0;
      const barWidth = Math.min(100, Math.round((att.abs_impact / maxVal) * 100));
      row.innerHTML = `
        <span class="att-name">${att.ticker}</span>
        <div class="att-track">
          <div class="att-fill" style="width: ${barWidth}%; background: ${isPos ? 'var(--green)' : 'var(--red)'};"></div>
        </div>
        <span class="att-val" style="color: ${isPos ? 'var(--green)' : 'var(--red)'};">
          ${isPos ? '+' : ''}${att.impact_pct.toFixed(3)}%
        </span>
      `;
      attributionList.appendChild(row);
    });
  }

  // -------------------------------------------------------------------------
  // 3. Real Holdout Equity Curve Visualizer (Actual Backtest Series)
  // -------------------------------------------------------------------------
  const canvas = document.getElementById("equityChart");
  if (canvas) {
    const ctx = canvas.getContext("2d");

    let holdoutDates = [];
    let holdoutSeries = {};
    let activeStrategies = ["buy_hold", "cash", "linear_reg", "decision_tree", "lightgbm", "elastic_net"];
    let hoverIndex = null;

    const STRATEGY_META = {
      buy_hold: { name: "Buy & Hold (VW)", color: "#f43f5e", dash: [5, 4], lw: 2.0 },
      cash: { name: "Cash Benchmark", color: "#94a3b8", dash: [3, 3], lw: 1.5 },
      linear_reg: { name: "Linear Regression (h=1)", color: "#10b981", dash: [], lw: 2.6 },
      decision_tree: { name: "Decision Tree (h=5)", color: "#a855f7", dash: [], lw: 2.6 },
      lightgbm: { name: "LightGBM (h=5)", color: "#f59e0b", dash: [], lw: 2.2 },
      elastic_net: { name: "Elastic Net (h=1)", color: "#38bdf8", dash: [], lw: 2.0 }
    };

    function loadEquitySeries() {
      fetch("/api/equity")
        .then(r => {
          if (!r.ok) throw new Error("Equity API not found");
          return r.json();
        })
        .then(resp => {
          if (resp && resp.equity && resp.equity.series) {
            holdoutDates = resp.equity.dates || [];
            holdoutSeries = resp.equity.series;
            resizeCanvas();
          }
        })
        .catch(() => {
          // Direct fallback to results/holdout_equity_series.json
          fetch("/results/holdout_equity_series.json")
            .then(r => r.json())
            .then(data => {
              if (data && data.series) {
                holdoutDates = data.dates || [];
                holdoutSeries = data.series;
                resizeCanvas();
              }
            })
            .catch(e => console.info("Holdout equity load notice:", e));
        });
    }

    function resizeCanvas() {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const w = Math.floor(rect.width);
      const h = Math.floor(rect.height);
      if (w <= 0 || h <= 0) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);

      drawChart();
    }

    function drawChart() {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      if (w <= 0 || h <= 0) return;

      ctx.clearRect(0, 0, w, h);

      if (!holdoutDates || holdoutDates.length === 0 || !holdoutSeries || Object.keys(holdoutSeries).length === 0) {
        ctx.fillStyle = "#64748b";
        ctx.font = "14px 'Plus Jakarta Sans', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Loading authentic holdout backtest equity trajectory...", w / 2, h / 2);
        ctx.textAlign = "left";
        return;
      }

      const padLeft = 70, padRight = 35, padTop = 30, padBottom = 42;
      const plotW = Math.max(10, w - padLeft - padRight);
      const plotH = Math.max(10, h - padTop - padBottom);

      const minVal = 650.0;
      const maxVal = 1350.0;

      // Background subtle grid
      ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
      ctx.lineWidth = 1;
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "right";

      for (let v = 700; v <= 1300; v += 150) {
        const y = padTop + plotH - ((v - minVal) / (maxVal - minVal)) * plotH;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();
        ctx.fillText(`€${v}`, padLeft - 10, y + 4);
      }

      // Initial Capital €1,000 Reference Line
      const y1000 = padTop + plotH - ((1000.0 - minVal) / (maxVal - minVal)) * plotH;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(padLeft, y1000);
      ctx.lineTo(w - padRight, y1000);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(56, 189, 248, 0.8)";
      ctx.fillText("Base €1,000", padLeft - 10, y1000 + 4);

      ctx.textAlign = "left";

      // Draw each active strategy curve
      const totalPoints = holdoutDates.length;

      activeStrategies.forEach(key => {
        const strat = holdoutSeries[key];
        const meta = STRATEGY_META[key] || { color: "#38bdf8", dash: [], lw: 2.0 };
        if (!strat || !strat.data || strat.data.length === 0) return;

        ctx.strokeStyle = meta.color;
        ctx.lineWidth = meta.lw || 2.0;
        ctx.setLineDash(meta.dash || []);

        ctx.beginPath();
        const dataArr = strat.data;
        const pts = dataArr.length;

        dataArr.forEach((val, i) => {
          const x = padLeft + (i / Math.max(pts - 1, 1)) * plotW;
          const y = padTop + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // Date labels on X-axis (sample 6 evenly spaced dates)
      ctx.fillStyle = "#64748b";
      ctx.font = "11px 'JetBrains Mono', monospace";
      const numLabels = 5;
      for (let i = 0; i <= numLabels; i++) {
        const idx = Math.min(Math.floor((i / numLabels) * (totalPoints - 1)), totalPoints - 1);
        const x = padLeft + (idx / (totalPoints - 1)) * plotW;
        const dStr = holdoutDates[idx] || "";
        ctx.fillText(dStr, Math.max(padLeft, x - 30), h - 14);
      }

      // Interactive Hover Crosshair & Tooltip
      if (hoverIndex !== null && hoverIndex >= 0 && hoverIndex < totalPoints) {
        const hX = padLeft + (hoverIndex / (totalPoints - 1)) * plotW;
        
        // Vertical dashed line
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(hX, padTop);
        ctx.lineTo(hX, padTop + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Tooltip box
        const dateStr = holdoutDates[hoverIndex] || "";
        const lines = [`Date: ${dateStr}`];
        activeStrategies.forEach(key => {
          const strat = holdoutSeries[key];
          if (strat && strat.data && hoverIndex < strat.data.length) {
            const val = strat.data[hoverIndex];
            const meta = STRATEGY_META[key] || {};
            const retPct = ((val - 1000.0) / 10.0).toFixed(1);
            lines.push(`${meta.name || key}: €${val.toFixed(1)} (${retPct > 0 ? '+' : ''}${retPct}%)`);
          }
        });

        const tipW = 210;
        const tipH = 20 + lines.length * 16;
        let tipX = hX + 15;
        if (tipX + tipW > w - 10) tipX = hX - tipW - 15;
        const tipY = padTop + 10;

        ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(tipX, tipY, tipW, tipH, 8) : ctx.rect(tipX, tipY, tipW, tipH);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#f8fafc";
        ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
        lines.forEach((l, li) => {
          if (li === 0) {
            ctx.fillStyle = "#38bdf8";
            ctx.font = "bold 11px 'JetBrains Mono', monospace";
          } else {
            ctx.fillStyle = "#cbd5e1";
            ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
          }
          ctx.fillText(l, tipX + 10, tipY + 16 + li * 16);
        });
      }
    }

    // Hover interactions
    canvas.addEventListener("mousemove", (e) => {
      if (!holdoutDates || holdoutDates.length === 0) return;
      const rect = canvas.getBoundingClientRect();
      const padLeft = 70, padRight = 35;
      const plotW = Math.max(10, rect.width - padLeft - padRight);
      const mouseX = e.clientX - rect.left;

      if (mouseX >= padLeft && mouseX <= rect.width - padRight) {
        const ratio = (mouseX - padLeft) / plotW;
        hoverIndex = Math.min(Math.floor(ratio * holdoutDates.length), holdoutDates.length - 1);
      } else {
        hoverIndex = null;
      }
      drawChart();
    });

    canvas.addEventListener("mouseleave", () => {
      hoverIndex = null;
      drawChart();
    });

    // Window & container resize observation
    window.addEventListener("resize", resizeCanvas);
    if (window.ResizeObserver) {
      new ResizeObserver(() => resizeCanvas()).observe(canvas.parentElement);
    }
    setTimeout(resizeCanvas, 50);

    loadEquitySeries();

    // Strategy toggles
    document.querySelectorAll(".strat-toggle").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.strat;
        if (activeStrategies.includes(key)) {
          if (activeStrategies.length > 1) {
            activeStrategies = activeStrategies.filter(k => k !== key);
            btn.classList.remove("active");
          }
        } else {
          activeStrategies.push(key);
          btn.classList.add("active");
        }
        drawChart();
      });
    });
  }

  // -------------------------------------------------------------------------
  // 4. Model Comparison Leaderboard (Dynamic from prediction_metrics.csv)
  // -------------------------------------------------------------------------
  const hFilterBtns = document.querySelectorAll(".h-filter-btn");
  const leaderboardBody = document.getElementById("leaderboardBody");
  let metricsRecords = [];

  function loadLeaderboardMetrics() {
    fetch("/api/metrics")
      .then(r => {
        if (!r.ok) throw new Error("Metrics API not found");
        return r.json();
      })
      .then(data => {
        if (data && data.metrics) {
          metricsRecords = data.metrics;
          renderLeaderboard(1);
        }
      })
      .catch(() => {
        // Fallback: fetch CSV directly
        fetch("/results/prediction_metrics.csv")
          .then(r => r.text())
          .then(csvText => parseAndRenderCSV(csvText))
          .catch(() => renderDefaultMetrics(1));
      });
  }

  function parseAndRenderCSV(csvText) {
    const lines = csvText.trim().split("\n");
    const headers = lines[0].split(",");
    const records = [];

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(",");
      if (parts.length >= headers.length) {
        const obj = {};
        headers.forEach((h, idx) => obj[h.trim()] = parts[idx].trim());
        if (obj.period === "hold") {
          records.push(obj);
        }
      }
    }
    metricsRecords = records;
    renderLeaderboard(1);
  }

  function renderLeaderboard(targetH) {
    if (!leaderboardBody) return;
    leaderboardBody.innerHTML = "";

    const filtered = metricsRecords.filter(r => parseInt(r.h) === targetH);

    if (filtered.length === 0) {
      renderDefaultMetrics(targetH);
      return;
    }

    // Sort by RMSE ascending
    filtered.sort((a, b) => parseFloat(a.RMSE_dec) - parseFloat(b.RMSE_dec));

    filtered.forEach(r => {
      const tr = document.createElement("tr");
      const isZero = r.Model === "Zero baseline";
      const isLight = r.Model === "LightGBM" && targetH === 1;
      const isEN = r.Model === "Elastic Net";

      let badge = "";
      if (isZero) badge = `<span class="tag-winner" style="background: rgba(148, 163, 184, 0.2); color: #94a3b8;">Benchmark</span>`;
      else if (isLight) badge = `<span class="tag-winner" style="background: rgba(52, 211, 153, 0.2); color: var(--green);">56.8% Dir. Hit</span>`;
      else if (isEN) badge = `<span class="tag-winner" style="background: rgba(56, 189, 248, 0.2); color: var(--primary);">Lowest ML Error</span>`;

      const rmseF = parseFloat(r.RMSE_dec).toFixed(4);
      const maeF = parseFloat(r.MAE_dec).toFixed(4);
      const r2F = parseFloat(r.OOS_R2).toFixed(4);
      const dirF = r.Direction_acc ? (parseFloat(r.Direction_acc) * 100).toFixed(1) + "%" : "–";
      const timeF = r.Time_s ? (parseFloat(r.Time_s) * 1000).toFixed(1) + " ms" : "0.0 ms";

      tr.innerHTML = `
        <td>
          <strong>${r.Model}</strong>
          ${badge}
        </td>
        <td><code>h = ${r.h}</code></td>
        <td><code>${rmseF}</code></td>
        <td><code>${maeF}</code></td>
        <td><code style="color: ${parseFloat(r2F) < 0 ? 'var(--red)' : 'inherit'};">${r2F}</code></td>
        <td><span style="color: ${dirF.startsWith('5') ? 'var(--green)' : 'inherit'};">${dirF}</span></td>
        <td><small style="color: var(--text-dim);">${timeF}</small></td>
      `;
      leaderboardBody.appendChild(tr);
    });
  }

  function renderDefaultMetrics(targetH) {
    const defaults = {
      1: [
        { model: "Zero baseline", h: 1, rmse: "0.0189", mae: "0.0141", r2: "0.0000", dir: "–", time: "0.0 ms", badge: "Benchmark" },
        { model: "Elastic Net", h: 1, rmse: "0.0189", mae: "0.0142", r2: "-0.0026", dir: "46.5%", time: "191.1 ms", badge: "Lowest ML Error" },
        { model: "LightGBM", h: 1, rmse: "0.0190", mae: "0.0142", r2: "-0.0070", dir: "56.8%", time: "411.2 ms", badge: "56.8% Dir. Hit" },
        { model: "Decision Tree", h: 1, rmse: "0.0191", mae: "0.0144", r2: "-0.0234", dir: "46.5%", time: "385.8 ms", badge: "" },
        { model: "XGBoost", h: 1, rmse: "0.0192", mae: "0.0144", r2: "-0.0272", dir: "50.2%", time: "786.0 ms", badge: "" },
        { model: "Linear Regression", h: 1, rmse: "0.0196", mae: "0.0149", r2: "-0.0768", dir: "45.8%", time: "39.9 ms", badge: "" }
      ],
      5: [
        { model: "Zero baseline", h: 5, rmse: "0.0393", mae: "0.0310", r2: "0.0000", dir: "–", time: "0.0 ms", badge: "Benchmark" },
        { model: "Elastic Net", h: 5, rmse: "0.0396", mae: "0.0312", r2: "-0.0160", dir: "44.9%", time: "189.2 ms", badge: "Lowest ML Error" },
        { model: "LightGBM", h: 5, rmse: "0.0400", mae: "0.0315", r2: "-0.0328", dir: "49.4%", time: "419.4 ms", badge: "" },
        { model: "Decision Tree", h: 5, rmse: "0.0401", mae: "0.0320", r2: "-0.0395", dir: "50.9%", time: "375.6 ms", badge: "" },
        { model: "XGBoost", h: 5, rmse: "0.0401", mae: "0.0316", r2: "-0.0396", dir: "46.1%", time: "911.3 ms", badge: "" },
        { model: "Linear Regression", h: 5, rmse: "0.0415", mae: "0.0329", r2: "-0.1149", dir: "47.6%", time: "38.3 ms", badge: "" }
      ],
      20: [
        { model: "Zero baseline", h: 20, rmse: "0.0756", mae: "0.0598", r2: "0.0000", dir: "–", time: "0.0 ms", badge: "Benchmark" },
        { model: "Elastic Net", h: 20, rmse: "0.0770", mae: "0.0607", r2: "-0.0365", dir: "44.8%", time: "193.3 ms", badge: "Lowest ML Error" },
        { model: "LightGBM", h: 20, rmse: "0.0781", mae: "0.0620", r2: "-0.0674", dir: "42.9%", time: "423.1 ms", badge: "" },
        { model: "XGBoost", h: 20, rmse: "0.0781", mae: "0.0622", r2: "-0.0680", dir: "47.2%", time: "783.9 ms", badge: "" },
        { model: "Linear Regression", h: 20, rmse: "0.0787", mae: "0.0628", r2: "-0.0827", dir: "50.8%", time: "38.1 ms", badge: "" },
        { model: "Decision Tree", h: 20, rmse: "0.0791", mae: "0.0628", r2: "-0.0942", dir: "44.0%", time: "399.4 ms", badge: "" }
      ]
    };

    const rows = defaults[targetH] || defaults[1];
    leaderboardBody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <strong>${r.model}</strong>
          ${r.badge ? `<span class="tag-winner">${r.badge}</span>` : ''}
        </td>
        <td><code>h = ${r.h}</code></td>
        <td><code>${r.rmse}</code></td>
        <td><code>${r.mae}</code></td>
        <td><code style="color: ${parseFloat(r.r2) < 0 ? 'var(--red)' : 'inherit'};">${r.r2}</code></td>
        <td><span style="color: ${r.dir.startsWith('5') ? 'var(--green)' : 'inherit'};">${r.dir}</span></td>
        <td><small style="color: var(--text-dim);">${r.time}</small></td>
      `;
      leaderboardBody.appendChild(tr);
    });
  }

  hFilterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      hFilterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderLeaderboard(parseInt(btn.dataset.h));
    });
  });

  // -------------------------------------------------------------------------
  // 5. Load Pipeline Summary Metadata (/api/summary)
  // -------------------------------------------------------------------------
  function loadPipelineSummary() {
    fetch("/api/summary")
      .then(r => r.json())
      .then(data => {
        const sum = data.summary || data;
        if (sum) {
          const elDays = document.getElementById("metaDays");
          const elFeats = document.getElementById("metaFeatures");
          const elPeriod = document.getElementById("metaPeriod");
          if (elDays && sum.total_trading_days) elDays.textContent = sum.total_trading_days.toLocaleString();
          if (elFeats && sum.feature_count) elFeats.textContent = sum.feature_count;
          if (elPeriod && sum.date_range) elPeriod.textContent = `Synchronized trading days (${sum.date_range[0]} to ${sum.date_range[1]})`;
        }
      })
      .catch(() => {});
  }

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------
  buildSliders();
  loadLeaderboardMetrics();
  loadPipelineSummary();

  // Initial model inference call
  triggerModelInference();
});
