/**
 * Explanation Modal Management
 * Handles AJAX calls to /explain-record endpoint and displays SHAP explanations
 */

// Show explanation modal
function showExplanationModal(recordData, rowIndex) {
  const modal = document.getElementById("explanation-modal");
  if (!modal) return;

  // Show loading state
  const content = modal.querySelector(".modal-body");
  content.innerHTML = '<div class="loading">Loading explanation...</div>';
  modal.style.display = "flex";

  // Prepare payload for /explain-record endpoint
  const payload = {
    record: recordData,
    index: rowIndex,
    dataset_context: document.body.dataset.datasetContext || "",
  };

  // Make AJAX request to /explain-record
  fetch("/explain-record", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }
      return response.json();
    })
    .then((data) => {
      displayExplanation(data);
    })
    .catch((error) => {
      content.innerHTML = `<div class="error-message">Failed to load explanation: ${error.message}</div>`;
      console.error("Explanation fetch error:", error);
    });
}

// Display explanation data in modal
function displayExplanation(data) {
  const modal = document.getElementById("explanation-modal");
  const content = modal.querySelector(".modal-body");

  let html = `
    <div class="explanation-header">
      <h3>Record Explanation</h3>
      ${data.index !== undefined ? `<p>Row Index: <strong>${data.index}</strong></p>` : ""}
    </div>

    <div class="explanation-section">
      <h4>Prediction Details</h4>
      <div class="metric-row">
        <span class="metric-label">Prediction Probability:</span>
        <span class="metric-value">${(data.prediction_probability * 100).toFixed(2)}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Base Value:</span>
        <span class="metric-value">${parseFloat(data.base_value).toFixed(4)}</span>
      </div>
    </div>

    <div class="explanation-section">
      <h4>Feature Contributions (SHAP)</h4>
      <div class="feature-contributions">
  `;

  // Display top contributing features
  if (data.feature_contributions && Object.keys(data.feature_contributions).length > 0) {
    const sortedFeatures = Object.entries(data.feature_contributions)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 10); // Show top 10 features

    sortedFeatures.forEach(([feature, contribution]) => {
      const impact = contribution > 0 ? "positive" : "negative";
      const barWidth = Math.abs(contribution) * 100; // Simplified visualization
      html += `
        <div class="feature-row">
          <span class="feature-name">${feature}</span>
          <div class="feature-bar-container">
            <div class="feature-bar ${impact}" style="width: ${Math.min(barWidth, 100)}%"></div>
          </div>
          <span class="feature-value">${contribution.toFixed(4)}</span>
        </div>
      `;
    });
  } else {
    html += `<p>No feature contributions available</p>`;
  }

  html += `
      </div>
    </div>
  `;

  // Add LLM explanation if available
  if (data.plain_english_explanation) {
    html += `
      <div class="explanation-section">
        <h4>Plain English Explanation</h4>
        <p>${data.plain_english_explanation}</p>
      </div>
    `;
  }

  content.innerHTML = html;
}

// Close modal
function closeExplanationModal() {
  const modal = document.getElementById("explanation-modal");
  if (modal) {
    modal.style.display = "none";
  }
}

// Attach click handlers to clickable rows
function initializeClickableRows() {
  const clickableRows = document.querySelectorAll("tr[data-record]");
  clickableRows.forEach((row) => {
    row.style.cursor = "pointer";
    row.addEventListener("click", function () {
      try {
        const recordData = JSON.parse(this.dataset.record);
        const rowIndex = this.dataset.index || 0;
        showExplanationModal(recordData, parseInt(rowIndex, 10));
      } catch (e) {
        console.error("Failed to parse record data:", e);
      }
    });

    // Visual feedback on hover
    row.addEventListener("mouseenter", function () {
      this.style.backgroundColor = "var(--surface-strong)";
    });
    row.addEventListener("mouseleave", function () {
      this.style.backgroundColor = "";
    });
  });
}

// Modal close button and backdrop handlers
document.addEventListener("DOMContentLoaded", function () {
  // Close on close button
  const closeBtn = document.querySelector(".modal-close");
  if (closeBtn) {
    closeBtn.addEventListener("click", closeExplanationModal);
  }

  // Close on backdrop click
  const modal = document.getElementById("explanation-modal");
  if (modal) {
    modal.addEventListener("click", function (event) {
      if (event.target === this) {
        closeExplanationModal();
      }
    });
  }

  // Initialize clickable rows
  initializeClickableRows();
});
