document.addEventListener("DOMContentLoaded", () => {

  const clock = document.getElementById("clock");
  function updateClock() {
    if (!clock) return;
    const now  = new Date();
    const hh   = String(now.getHours()).padStart(2, "0");
    const mm   = String(now.getMinutes()).padStart(2, "0");
    const ss   = String(now.getSeconds()).padStart(2, "0");
    const date = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
    clock.textContent = `${date}  ${hh}:${mm}:${ss}`;
  }
  updateClock();
  setInterval(updateClock, 1000);

  const form       = document.getElementById("loanForm");
  const submitBtn  = document.getElementById("submitBtn");
  const btnInner   = document.getElementById("btnInner");
  const btnLoading = document.getElementById("btnLoading");

  if (form) {
    form.addEventListener("submit", () => {
      if (btnInner)   btnInner.style.display  = "none";
      if (btnLoading) btnLoading.style.display = "flex";
      if (submitBtn)  submitBtn.disabled       = true;
    });
  }

  const creditInput = document.getElementById("cibil_score");
  const scoreBar    = document.getElementById("scoreBar");
  const scoreBadge  = document.getElementById("scoreBadge");

  function getScoreStyle(val) {
    if (val < 500) return { label: "Poor",       color: "#8b1a1a", bg: "#fde8e8" };
    if (val < 600) return { label: "Fair",        color: "#92400e", bg: "#fef3e2" };
    if (val < 700) return { label: "Good",        color: "#1a6e47", bg: "#edf7f2" };
    if (val < 800) return { label: "Very Good",   color: "#1a4a8b", bg: "#e8f0fd" };
    return              { label: "Exceptional",   color: "#5b21b6", bg: "#f3e8ff" };
  }

  function updateScoreBar(val) {
    if (!scoreBar || !scoreBadge) return;
    if (!val || isNaN(val) || val < 300 || val > 900) {
      scoreBar.style.width      = "0%";
      scoreBar.style.background = "#ddd";
      scoreBadge.textContent    = "";
      scoreBadge.style.cssText  = "";
      return;
    }
    const pct   = ((val - 300) / (900 - 300)) * 100;
    const style = getScoreStyle(val);
    scoreBar.style.width           = pct + "%";
    scoreBar.style.backgroundColor = style.color;
    scoreBadge.textContent         = style.label;
    scoreBadge.style.color         = style.color;
    scoreBadge.style.background    = style.bg;
    scoreBadge.style.padding       = "0.1rem 0.45rem";
    scoreBadge.style.borderRadius  = "2px";
    scoreBadge.style.fontSize      = "0.62rem";
    scoreBadge.style.fontFamily    = "var(--ff-mono)";
    scoreBadge.style.fontWeight    = "500";
  }

  if (creditInput) {
    creditInput.addEventListener("input", () => {
      updateScoreBar(parseInt(creditInput.value, 10));
    });
    if (creditInput.value) updateScoreBar(parseInt(creditInput.value, 10));
  }

  const confFill = document.querySelector(".conf-fill");
  if (confFill) {
    const target = confFill.getAttribute("data-confidence");
    setTimeout(() => {
      confFill.style.width = target + "%";
    }, 100);
  }

  const resultWrap = document.getElementById("resultWrap");
  if (resultWrap) {
    setTimeout(() => {
      resultWrap.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 200);
  }

});