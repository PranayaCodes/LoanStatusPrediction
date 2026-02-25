// script.js — LoanIQ Credit Assessment Terminal

document.addEventListener("DOMContentLoaded", () => {

  // ── Live Clock ──────────────────────────────────────────
  const clock = document.getElementById("clock");
  function updateClock() {
    if (!clock) return;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    const ss = String(now.getSeconds()).padStart(2, "0");
    const dd = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
    clock.textContent = `${dd}  ${hh}:${mm}:${ss}`;
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ── Submit Loading State ────────────────────────────────
  const form      = document.getElementById("loanForm");
  const submitBtn = document.getElementById("submitBtn");
  const btnInner  = submitBtn?.querySelector(".btn-inner");
  const btnLoading = document.getElementById("btnLoading");

  if (form) {
    form.addEventListener("submit", () => {
      if (btnInner)  btnInner.style.display  = "none";
      if (btnLoading) btnLoading.style.display = "block";
      if (submitBtn)  submitBtn.disabled = true;
    });
  }

  // ── Credit Score Bar & Badge ────────────────────────────
  const creditInput = document.getElementById("credit_score");
  const scoreBar    = document.getElementById("scoreBar");
  const scoreBadge  = document.getElementById("scoreBadge");

  function updateScoreVisuals(val) {
    if (!scoreBar || !scoreBadge) return;

    if (!val || isNaN(val) || val < 300) {
      scoreBar.style.width = "0%";
      scoreBadge.textContent = "";
      scoreBadge.style.background = "transparent";
      return;
    }

    const pct = Math.min(((val - 300) / (850 - 300)) * 100, 100);
    scoreBar.style.width = pct + "%";

    let label, color, bg;

    if (val < 580) {
      label = "Poor"; color = "#8b1a1a"; bg = "#fde8e8";
    } else if (val < 670) {
      label = "Fair"; color = "#7a4a00"; bg = "#fef3e2";
    } else if (val < 740) {
      label = "Good"; color = "#1a6e47"; bg = "#edf7f2";
    } else if (val < 800) {
      label = "Very Good"; color = "#1a4a8b"; bg = "#e8f0fd";
    } else {
      label = "Exceptional"; color = "#1a4a8b"; bg = "#dce8ff";
    }

    scoreBar.style.background = color;
    scoreBadge.textContent = label;
    scoreBadge.style.color = color;
    scoreBadge.style.background = bg;
  }

  if (creditInput) {
    creditInput.addEventListener("input", () => {
      updateScoreVisuals(parseInt(creditInput.value, 10));
    });
    // Init on page load if value exists
    if (creditInput.value) {
      updateScoreVisuals(parseInt(creditInput.value, 10));
    }
  }

  // ── Scroll to result ────────────────────────────────────
  const resultWrap = document.getElementById("resultWrap");
  if (resultWrap) {
    setTimeout(() => {
      resultWrap.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 150);
  }

});