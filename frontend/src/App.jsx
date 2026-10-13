import { useEffect, useState } from "react";

import {
  Upload,
  Database,
  Brain,
  BarChart3,
  AlertTriangle,
  Lightbulb,
  FlaskConical,
} from "lucide-react";

import {
  checkBackend,
  uploadDataset,
  trainModel,
  getErrors,
  getReport,
} from "./api";

import "./App.css";

function App() {
  // ============================================================
  // STATE
  // ============================================================

  const [activeSection, setActiveSection] = useState("overview");

  const [dataset, setDataset] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [trainError, setTrainError] = useState("");

  const [errorResult, setErrorResult] = useState(null);
  const [errorLoading, setErrorLoading] = useState(false);

  const [reportResult, setReportResult] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);

  const [backendConnected, setBackendConnected] = useState(false);

  // ============================================================
  // BACKEND CONNECTION
  // ============================================================

  useEffect(() => {
    checkBackend()
      .then((data) => {
        console.log("Backend connected:", data);
        setBackendConnected(true);
      })
      .catch((error) => {
        console.error("Backend connection failed:", error);
        setBackendConnected(false);
      });
  }, []);

  // ============================================================
  // NAVIGATION
  // ============================================================

  const menuItems = [
    {
      id: "overview",
      label: "Overview",
      icon: Database,
    },
    {
      id: "quality",
      label: "Data Quality",
      icon: AlertTriangle,
    },
    {
      id: "model",
      label: "Model Performance",
      icon: Brain,
    },
    {
      id: "features",
      label: "Feature Importance",
      icon: BarChart3,
    },
    {
      id: "explain",
      label: "Explainability",
      icon: Lightbulb,
    },
    {
      id: "whatif",
      label: "What-If Simulator",
      icon: FlaskConical,
    },
  ];

  // ============================================================
  // UPLOAD DATASET
  // ============================================================

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    try {
      setUploading(true);
      setUploadError("");

      // Reset previous analysis
      setTrainingResult(null);
      setTrainError("");
      setErrorResult(null);
      setReportResult(null);

      const result = await uploadDataset(file);

      console.log("Upload result:", result);

      setDataset(result);
      setActiveSection("overview");
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadError(
        error?.response?.data?.detail ||
          "Failed to upload dataset."
      );
    } finally {
      setUploading(false);

      // Allow selecting the same file again
      event.target.value = "";
    }
  };

// ============================================================
// TRAIN MODEL
// ============================================================

const handleTrainModel = async () => {
  if (!dataset?.filename) {
    return;
  }

  try {
    setTraining(true);
    setTrainError("");

    // Reset previous analysis
    setErrorResult(null);
    setReportResult(null);

    // --------------------------------------------------------
    // STEP 1: Train model
    // --------------------------------------------------------

    const result = await trainModel(dataset.filename);

    console.log("Training result:", result);

    setTrainingResult(result);

    // --------------------------------------------------------
    // STEP 2: Error analysis
    // --------------------------------------------------------

    setErrorLoading(true);

    try {
      const errors = await getErrors(dataset.filename);

      console.log("Error analysis result:", errors);

      setErrorResult(errors);
    } catch (error) {
    console.error("Training failed:", error);

    setTrainError(
      error?.response?.data?.detail ||
        "Model training failed."
    );
  } finally {
    setTraining(false);
  };


    // --------------------------------------------------------
    // STEP 3: Generate investigation report
    // --------------------------------------------------------

    setReportLoading(true);

    try {
      const report = await getReport(
        dataset.filename
      );

      console.log(
  "Investigation report:",
  report
);

console.log(
  "Feature analysis:",
  report?.report?.feature_analysis
);

      setReportResult(report);
    } catch (error) {
      console.error(
        "Report generation failed:",
        error
      );
    } finally {
      setReportLoading(false);
    }

  } catch (error) {
    console.error(
      "Training failed:",
      error
    );

    setTrainError(
      error?.response?.data?.detail ||
        "Model training failed."
    );
  } finally {
    setTraining(false);
  }
};

  

  // ============================================================
  // HELPER FUNCTIONS
  // ============================================================

  const getBestModelMetrics = () => {
    if (!trainingResult?.training) {
      return null;
    }

    const bestModel = trainingResult.training.best_model;

    return (
      trainingResult.training.results?.[bestModel] || null
    );
  };

  const bestModelMetrics = getBestModelMetrics();

  // ------------------------------------------------------------
  // Outlier helpers
  // ------------------------------------------------------------

  const outliers = dataset?.investigation?.outliers || {};

  const iqrOutliers = outliers.iqr || {};
  const zscoreOutliers = outliers.zscore || {};

  const getOutlierTotal = (data) => {
  if (!data || typeof data !== "object") {
    return 0;
  }

  return Object.values(data).reduce((total, value) => {
    if (typeof value === "object" && value !== null) {
      return total + Number(value.outlier_count || 0);
    }

    return total + Number(value || 0);
  }, 0);
};

  const totalIQROutliers = getOutlierTotal(iqrOutliers);
  const totalZScoreOutliers = getOutlierTotal(zscoreOutliers);

  const isolationForestData =
    outliers.isolation_forest || [];

  const isolationForestCount = Array.isArray(
    isolationForestData
  )
    ? isolationForestData.length
    : typeof isolationForestData === "number"
    ? isolationForestData
    : 0;

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside className="sidebar">

        {/* Logo */}
        <div className="logo">
          <Brain size={28} />

          <div>
            <h2>AI Data Scientist</h2>
            <span>Automated ML Investigator</span>
          </div>
        </div>

        {/* Navigation */}
        <nav>
          {menuItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={`nav-item ${
                  activeSection === item.id
                    ? "active"
                    : ""
                }`}
                onClick={() =>
                  setActiveSection(item.id)
                }
              >
                <Icon size={19} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Upload */}
        <div className="sidebar-bottom">

          <label className="upload-button">
            <Upload size={18} />

            {uploading
              ? "Uploading..."
              : "Upload Dataset"}

            <input
              type="file"
              accept=".csv"
              hidden
              onChange={handleUpload}
            />
          </label>

          {uploadError && (
            <p className="error-message">
              {uploadError}
            </p>
          )}

        </div>
      </aside>

      {/* ======================================================
          MAIN CONTENT
      ====================================================== */}

      <main className="main-content">

        {/* ====================================================
            TOP BAR
        ==================================================== */}

        <header className="topbar">

          <div>
            <h1>AI Data Scientist</h1>

            <p>
              Intelligent automated dataset investigation
              and machine learning
            </p>
          </div>

          <div className="status">

            <span
              className={`status-dot ${
                backendConnected
                  ? ""
                  : "status-offline"
              }`}
            ></span>

            {backendConnected
              ? "Backend Connected"
              : "Backend Offline"}

          </div>

        </header>

        {/* ====================================================
            PAGE CONTENT
        ==================================================== */}

        <section className="content">

          {/* ==================================================
              PAGE HEADING
          ================================================== */}

          <div className="page-heading">

            <div>
              <h2>
                {
                  menuItems.find(
                    (item) => item.id === activeSection
                  )?.label
                }
              </h2>

              <p>
                Analyze your dataset and discover
                machine learning insights.
              </p>
            </div>

            {dataset?.filename && (
              <button
                className="train-button"
                disabled={training}
                onClick={handleTrainModel}
              >
                <Brain size={18} />

                {training
                  ? "Training..."
                  : "Train Model"}
              </button>
            )}

          </div>

          {/* Training error */}
          {trainError && (
            <div className="error-message">
              {trainError}
            </div>
          )}

          {/* ==================================================
              DATASET SUMMARY CARDS
          ================================================== */}

          <div className="stats-grid">

            {/* Dataset */}
            <div className="stat-card">

              <Database size={24} />

              <div>
                <span>Dataset</span>

                <strong>
                  {dataset?.filename ||
                    "No dataset"}
                </strong>
              </div>

            </div>

            {/* Target */}
            <div className="stat-card">

              <Brain size={24} />

              <div>
                <span>Target</span>

                <strong>
                  {dataset
                    ?.target_analysis
                    ?.selected_target ||
                    "Not detected"}
                </strong>
              </div>

            </div>

            {/* Problem Type */}
            <div className="stat-card">

              <BarChart3 size={24} />

              <div>
                <span>Problem Type</span>

                <strong>
                  {dataset
                    ?.target_analysis
                    ?.problem
                    ?.problem_type ||
                    "Not detected"}
                </strong>
              </div>

            </div>

            {/* Best Model */}
            <div className="stat-card">

              <Lightbulb size={24} />

              <div>
                <span>Best Model</span>

                <strong>
                  {trainingResult
                    ?.training
                    ?.best_model ||
                    "Not trained"}
                </strong>
              </div>

            </div>

          </div>

          {/* ==================================================
              OVERVIEW SECTION
          ================================================== */}

          {activeSection === "overview" && (
            <>
              {/* AI Investigation */}
              <div className="dashboard-panel">

                <div className="panel-header">

                  <h3>AI Investigation</h3>

                  <span className="badge">
                    {dataset
                      ? "Analysis Ready"
                      : "Waiting for Dataset"}
                  </span>

                </div>

                <div className="investigation-grid">

                  {/* CV Accuracy */}
                  <div className="investigation-item">

                    <span>
                      Cross-validation Accuracy
                    </span>

                    <strong>
                      {bestModelMetrics
                        ?.accuracy_mean !==
                      undefined
                        ? `${(
                            bestModelMetrics.accuracy_mean *
                            100
                          ).toFixed(2)}%`
                        : "—"}
                    </strong>

                  </div>

                  {/* CV F1 */}
                  <div className="investigation-item">

                    <span>
                      Cross-validation F1
                    </span>

                    <strong>
                      {bestModelMetrics
                        ?.f1_mean !== undefined
                        ? `${(
                            bestModelMetrics.f1_mean *
                            100
                          ).toFixed(2)}%`
                        : "—"}
                    </strong>

                  </div>

                  {/* Rows */}
                  <div className="investigation-item">

                    <span>Analyzed Rows</span>

                    <strong>
                      {dataset
                        ?.investigation
                        ?.dataset
                        ?.rows ?? "—"}
                    </strong>

                  </div>

                  {/* Errors */}
                  <div className="investigation-item">

                    <span>Model Errors</span>

                    <strong>
                      {errorLoading
                        ? "..."
                        : errorResult
                            ?.error_analysis
                            ?.incorrect_predictions ??
                          "—"}
                    </strong>

                  </div>

                </div>

              </div>

              {/* Overview AI Insight */}
              <div className="insight-card">

                <Lightbulb size={24} />

                <div>

                  <h3>AI Insight</h3>

                  <p>
                    {trainingResult
                      ? "The AI Data Scientist has completed model training and dataset investigation. Explore the sections from the sidebar for detailed analysis."
                      : "Upload a dataset and train a model to generate automated machine learning insights."}
                  </p>

                </div>

              </div>
            </>
          )}

          {/* ==================================================
              DATA QUALITY
          ================================================== */}

          {activeSection === "quality" && dataset && (
            <div className="dashboard-panel">

              <div className="panel-header">

                <div>

                  <h3>
                    Data Quality Analysis
                  </h3>

                  <p className="panel-subtitle">
                    Automated investigation of
                    dataset quality
                  </p>

                </div>

                <span className="badge">
                  Health Score:{" "}
                  {dataset
                    .investigation
                    ?.quality
                    ?.health_score ??
                    "N/A"}
                </span>

              </div>

              {/* Quality summary cards */}
              <div className="quality-grid">

                {/* Dataset Size */}
                <div className="quality-card">

                  <Database size={22} />

                  <span>Dataset Size</span>

                  <strong>
                    {dataset
                      .investigation
                      ?.dataset
                      ?.rows ?? 0}{" "}
                    rows ×{" "}
                    {dataset
                      .investigation
                      ?.dataset
                      ?.columns ?? 0}{" "}
                    columns
                  </strong>

                </div>

                {/* Duplicate Rows */}
                <div className="quality-card">

                  <AlertTriangle size={22} />

                  <span>Duplicate Rows</span>

                  <strong>
                    {dataset
                      .investigation
                      ?.quality
                      ?.duplicates
                      ?.count ?? 0}
                  </strong>

                </div>

                {/* Missing Values */}
                <div className="quality-card">

                  <AlertTriangle size={22} />

                  <span>Missing Values</span>

                  <strong>
                    {
                      Object.keys(
                        dataset
                          .investigation
                          ?.quality
                          ?.missing_values || {}
                      ).length
                    }
                  </strong>

                </div>

                {/* Constant Columns */}
                <div className="quality-card">

                  <AlertTriangle size={22} />

                  <span>Constant Columns</span>

                  <strong>
                    {dataset
                      .investigation
                      ?.quality
                      ?.constant_columns
                      ?.length ?? 0}
                  </strong>

                </div>

              </div>

              {/* ==================================================
                  CONSTANT COLUMNS
              ================================================== */}

              {dataset
                .investigation
                ?.quality
                ?.constant_columns
                ?.length > 0 && (

                <div className="quality-detail">

                  <h4>
                    Constant Columns
                  </h4>

                  <div className="tag-container">

                    {dataset.investigation.quality.constant_columns.map(
                      (column) => (
                        <span
                          className="data-tag"
                          key={column}
                        >
                          {column}
                        </span>
                      )
                    )}

                  </div>

                </div>
              )}

              {/* ==================================================
                  HIGH CARDINALITY
              ================================================== */}

              {Object.keys(
                dataset
                  .investigation
                  ?.quality
                  ?.high_cardinality || {}
              ).length > 0 && (

                <div className="quality-detail">

                  <h4>
                    High Cardinality Columns
                  </h4>

                  {Object.entries(
                    dataset.investigation.quality
                      .high_cardinality
                  ).map(
                    ([column, info]) => (
                      <div
                        className="quality-row"
                        key={column}
                      >

                        <span>
                          {column}
                        </span>

                        <strong>
                          {info.unique_values}{" "}
                          unique values (
                          {info.percentage}%)
                        </strong>

                      </div>
                    )
                  )}

                </div>
              )}

              {/* ==================================================
                  OUTLIER DETECTION
              ================================================== */}

              <div className="quality-detail">

                <h4>Outlier Detection</h4>

                {/* IQR */}
                <div className="quality-row">
                  <span>IQR Method</span>

                  <strong>
                    {totalIQROutliers} outliers detected
                  </strong>
                </div>

                {/* Z-Score */}
                <div className="quality-row">
                  <span>Z-Score Method</span>

                  <strong>
                    {totalZScoreOutliers} outliers detected
                  </strong>
                </div>

                {/* Isolation Forest */}
                <div className="quality-row">
                  <span>Isolation Forest</span>

                  <strong>
                    {isolationForestCount} anomalies detected
                  </strong>
                </div>

                {/* IQR Details */}
                {Object.keys(iqrOutliers).length > 0 && (
                  <div className="quality-detail">

                    <h4>IQR Details</h4>

                    {Object.entries(iqrOutliers).map(
                      ([column, info]) => (
                        <div
                          className="quality-row"
                          key={`iqr-${column}`}
                        >
                          <span>{column}</span>

                          <strong>
                            {typeof info === "object"
                              ? `${info.outlier_count ?? 0} outliers (${info.outlier_percentage ?? 0}%)`
                              : `${info} outliers`}
                          </strong>
                        </div>
                      )
                    )}

                  </div>
                )}

                {/* Z-Score Details */}
                {Object.keys(zscoreOutliers).length > 0 && (
                  <div className="quality-detail">

                    <h4>Z-Score Details</h4>

                    {Object.entries(zscoreOutliers).map(
                      ([column, info]) => (
                        <div
                          className="quality-row"
                          key={`zscore-${column}`}
                        >
                          <span>{column}</span>

                          <strong>
                            {typeof info === "object"
                              ? `${info.outlier_count ?? 0} outliers (${info.outlier_percentage ?? 0}%)`
                              : `${info} outliers`}
                          </strong>
                        </div>
                      )
                    )}

                  </div>
                )}

              </div>

            </div>
          )}

          {/* ==================================================
              NO DATA MESSAGE FOR QUALITY
          ================================================== */}

          {activeSection === "quality" &&
            !dataset && (

              <div className="dashboard-panel">

                <div className="empty-state">

                  <Database size={40} />

                  <h3>
                    No Dataset Available
                  </h3>

                  <p>
                    Upload a CSV dataset to
                    perform automatic data
                    quality analysis.
                  </p>

                </div>

              </div>
            )}

          {/* ==================================================
              MODEL PERFORMANCE
          ================================================== */}

          {activeSection === "model" && (
            <>
              {trainingResult ? (

                <div className="dashboard-panel">

                  <div className="panel-header">

                    <div>

                      <h3>
                        Model Performance
                      </h3>

                      <p className="panel-subtitle">
                        Comparison of trained
                        machine learning models
                      </p>

                    </div>

                    <span className="badge">
                      Best:{" "}
                      {
                        trainingResult
                          .training
                          .best_model
                      }
                    </span>

                  </div>

                  <div className="model-performance-grid">

                    {Object.entries(
                      trainingResult
                        .training
                        .results || {}
                    ).map(
                      ([modelName, metrics]) => (

                        <div
                          className="model-card"
                          key={modelName}
                        >

                          <h4>
                            {modelName}
                          </h4>

                          {/* CV Accuracy */}
                          <div className="metric-row">

                            <span>
                              CV Accuracy
                            </span>

                            <strong>
                              {metrics.accuracy_mean !==
                              undefined
                                ? `${(
                                    metrics.accuracy_mean *
                                    100
                                  ).toFixed(2)}%`
                                : "N/A"}
                            </strong>

                          </div>

                          {/* CV F1 */}
                          <div className="metric-row">

                            <span>
                              CV F1 Score
                            </span>

                            <strong>
                              {metrics.f1_mean !==
                              undefined
                                ? `${(
                                    metrics.f1_mean *
                                    100
                                  ).toFixed(2)}%`
                                : "N/A"}
                            </strong>

                          </div>

                          {/* Precision */}
                          <div className="metric-row">

                            <span>
                              Precision
                            </span>

                            <strong>
                              {metrics.precision_mean !==
                              undefined
                                ? `${(
                                    metrics.precision_mean *
                                    100
                                  ).toFixed(2)}%`
                                : "N/A"}
                            </strong>

                          </div>

                          {/* Recall */}
                          <div className="metric-row">

                            <span>
                              Recall
                            </span>

                            <strong>
                              {metrics.recall_mean !==
                              undefined
                                ? `${(
                                    metrics.recall_mean *
                                    100
                                  ).toFixed(2)}%`
                                : "N/A"}
                            </strong>

                          </div>

                          {/* Holdout Accuracy */}
                          <div className="metric-row">

                            <span>
                              Holdout Accuracy
                            </span>

                            <strong>
                              {metrics.holdout_accuracy !==
                              undefined
                                ? `${(
                                    metrics.holdout_accuracy *
                                    100
                                  ).toFixed(2)}%`
                                : "N/A"}
                            </strong>

                          </div>

                          {/* Holdout F1 */}
                          <div className="metric-row">

                            <span>
                              Holdout F1
                            </span>

                            <strong>
                              {metrics.holdout_f1 !==
                              undefined
                                ? `${(
                                    metrics.holdout_f1 *
                                    100
                                  ).toFixed(2)}%`
                                : "N/A"}
                            </strong>

                          </div>

                        </div>
                      )
                    )}

                  </div>

                </div>

              ) : (

                <div className="dashboard-panel">

                  <div className="empty-state">

                    <Brain size={40} />

                    <h3>
                      Model Not Trained
                    </h3>

                    <p>
                      Upload a dataset and click
                      "Train Model" to compare
                      machine learning models.
                    </p>

                  </div>

                </div>

              )}
            </>
          )}

             {/* ==================================================
                FEATURE IMPORTANCE
              ================================================== */}

            {activeSection === "features" && (
              <div className="dashboard-panel">

                <div className="panel-header">

                  <div>
                    <h3>Feature Importance</h3>

                    <p className="panel-subtitle">
                      Features influencing the selected model
                    </p>
                  </div>

                  <span className="badge">
                    Method:{" "}
                    {reportResult?.report?.feature_analysis?.method ||
                      "N/A"}
                  </span>

                </div>

                {reportLoading ? (
                  <div className="empty-state">
                    <BarChart3 size={40} />

                    <h3>Analyzing Features...</h3>

                    <p>
                      The AI Data Scientist is calculating
                      feature importance.
                    </p>
                  </div>

                ) : reportResult?.report?.feature_analysis
                    ?.top_features?.length > 0 ? (

                  <div className="feature-importance-list">

                    {reportResult.report.feature_analysis.top_features.map(
                      (feature, index) => {

                        const rawFeatureName =
                        feature.feature ||
                        feature.name ||
                        feature.column ||
                        `Feature ${index + 1}`;

                      const featureName = rawFeatureName
                        .replace(/^numerical__/, "")
                        .replace(/^categorical__/, "")
                        .replaceAll("_", " ");

                        const importance = Number(
                        feature.importance ??
                        feature.value ??
                        feature.score ??
                        0
                      );

                      const maxImportance = Math.max(
                        ...reportResult.report.feature_analysis.top_features.map(
                          (item) =>
                            Number(
                              item.importance ??
                              item.value ??
                              item.score ??
                              0
                            )
                        ),
                        1
                      );

                      const barWidth = (importance / maxImportance) * 100;

                        return (
                          
                        <div className="feature-importance-item" key={featureName}>
                          <div className="feature-header">
                            <div className="feature-name">
                              <span className="feature-rank">#{index + 1}</span>
                              <strong title={featureName}>{featureName}</strong>
                            </div>

                            <span className="feature-value">
                              {importance.toFixed(4)}
                            </span>
                          </div>

                          <div className="feature-bar-background">
                            <div
                              className="feature-bar"
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                        </div>

                        );
                      }
                    )}

                  </div>

                ) : (

                  <div className="empty-state">

                    <BarChart3 size={40} />

                    <h3>
                      Feature Importance Not Available
                    </h3>

                    <p>
                      Train a model first to calculate
                      feature importance.
                    </p>

                  </div>

                )}

              </div>
            )}

          {/* ==================================================
              EXPLAINABILITY
          ================================================== */}

          {activeSection === "explain" && (

            <div className="dashboard-panel">

              <div className="panel-header">

                <div>

                  <h3>
                    Explainability
                  </h3>

                  <p className="panel-subtitle">
                    Understand why the model
                    makes its predictions
                  </p>

                </div>

              </div>

              <div className="empty-state">

                <Lightbulb size={40} />

                <h3>
                  Explainable AI
                </h3>

                <p>
                  SHAP-based prediction
                  explanations will appear here.
                </p>

              </div>

            </div>
          )}

          {/* ==================================================
              WHAT-IF SIMULATOR
          ================================================== */}

          {activeSection === "whatif" && (

            <div className="dashboard-panel">

              <div className="panel-header">

                <div>

                  <h3>
                    What-If Simulator
                  </h3>

                  <p className="panel-subtitle">
                    Change a feature and observe
                    how the prediction changes
                  </p>

                </div>

              </div>

              <div className="empty-state">

                <FlaskConical size={40} />

                <h3>
                  Interactive What-If Analysis
                </h3>

                <p>
                  Select a feature and modify its
                  value to compare the original
                  and modified predictions.
                </p>

              </div>

            </div>
          )}

          {/* ==================================================
              DATASET NOT AVAILABLE
          ================================================== */}

          {!dataset &&
            activeSection !== "overview" &&
            activeSection !== "quality" && (
              <div className="dashboard-panel">

                <div className="empty-state">

                  <Database size={40} />

                  <h3>
                    Upload a Dataset
                  </h3>

                  <p>
                    Upload a CSV dataset from the
                    sidebar to start the AI Data
                    Scientist investigation.
                  </p>

                </div>

              </div>
            )}

        </section>
      </main>
    </div>
  );
}

export default App;