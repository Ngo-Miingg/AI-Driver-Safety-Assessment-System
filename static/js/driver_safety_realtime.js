document.addEventListener("DOMContentLoaded", () => {
  const shell = document.querySelector(".driver-safety-console");
  const socket = window.io ? window.io() : null;
  const start = document.getElementById("driver-safety-start");
  const stop = document.getElementById("driver-safety-stop");
  const feed = document.getElementById("driver-safety-feed");
  const placeholder = document.getElementById("camera-placeholder");
  const error = document.getElementById("driver-safety-error");
  const connection = document.getElementById("connection-state");
  const status = document.getElementById("live-status");
  const modelReady = document.getElementById("model-ready");
  const faceStatePill = document.getElementById("face-state-pill");
  // Standalone phone detection is disabled; phone interaction is classified
  // by the driver action model and remains part of the model payload.
  const phoneBackend = "disabled";
  let running = false;

  const t = (key, fallback) => (window.I18N && window.I18N[key]) || fallback;
  const set = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
  const showError = (message) => { error.textContent = message; error.classList.remove("hidden"); };
  const setCardState = (id, tone) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("is-good", "is-warn", "is-danger", "is-neutral");
    el.classList.add(tone || "is-neutral");
  };
  const setBadgeState = (el, text, tone) => {
    if (!el) return;
    el.textContent = text;
    el.className = `badge ${tone || "badge-neutral"}`;
  };
  const actionText = (label) => {
    const labels = {
      normal_driving: t("driver_v1.action_normal", "Safe driving"),
      drowsiness: t("driver_v1.event_drowsiness", "Drowsiness"),
      phone_usage: t("driver_v1.event_phone_usage", "Phone usage"),
      phone_call: t("driver_v1.event_phone_call", "Phone call"),
      phone_texting: t("driver_v1.action_texting", "Texting"),
      phone_calling: t("driver_v1.action_calling", "Calling"),
      operating_radio: t("driver_v1.action_radio", "Operating radio"),
      drinking_eating: t("driver_v1.action_drinking_eating", "Drinking/eating"),
      reaching_behind: t("driver_v1.action_reaching", "Reaching behind"),
      hair_and_makeup: t("driver_v1.action_makeup", "Hair and makeup"),
      yawn: t("driver_v1.yawning", "Yawning"),
      looking_away: t("driver_v1.looking_away", "Looking away"),
      head_down: t("driver_v1.head_down", "Head down"),
    };
    return labels[label] || label || "--";
  };
  const renderActionProbabilities = (probabilities) => {
    const wrap = document.getElementById("driver-action-probabilities");
    if (!wrap) return;
    if (!probabilities || Object.keys(probabilities).length === 0) {
      wrap.innerHTML = `<p class="muted">${t("driver_v1.action_collecting", "Collecting a 16-frame clip.")}</p>`;
      return;
    }
    const rows = Object.entries(probabilities)
      .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
      .map(([label, value]) => {
        const pct = Math.min(100, Math.max(0, Number(value || 0) * 100));
        return `<div class="driver-action-probability-row">
          <span>${actionText(label)}</span>
          <div class="driver-action-probability-bar"><i style="width:${pct.toFixed(1)}%"></i></div>
          <strong>${Number(value || 0).toFixed(3)}</strong>
        </div>`;
      }).join("");
    wrap.innerHTML = rows;
  };
  const setRunning = (next) => {
    running = next;
    start.disabled = next;
    stop.disabled = !next;
    status.textContent = next ? t("driver_v1.running", "Monitoring") : t("driver_v1.waiting", "Waiting");
    status.className = `badge ${next ? "badge-success" : "badge-neutral"}`;
  };
  const setFeed = (enabled) => {
    if (!enabled) { feed.removeAttribute("src"); placeholder.classList.remove("hidden"); return; }
    feed.src = `${shell.dataset.feedUrl}?ts=${Date.now()}`;
    placeholder.classList.add("hidden");
  };

  if (!socket) { showError(t("driver_v1.socket_missing", "Could not load the Socket.IO client.")); return; }
  socket.on("connect", () => { connection.textContent = t("driver_v1.connected", "Connected"); connection.className = "badge badge-success"; });
  socket.on("disconnect", () => { connection.textContent = t("driver_v1.disconnected", "Disconnected"); connection.className = "badge badge-warning"; setRunning(false); setFeed(false); });
  socket.on("realtime_status", (data) => { if (data.status === "running") { setRunning(true); setFeed(true); } });
  socket.on("realtime_error", (data) => { showError(data.message || t("driver_v1.pipeline_failed", "The model pipeline could not run.")); setRunning(false); setFeed(false); });
  socket.on("realtime_completed", (data) => { setRunning(false); setFeed(false); if (data.session_url) { const link = document.getElementById("session-link"); link.href = data.session_url; link.classList.remove("hidden"); } });
  socket.on("realtime_update", (data) => {
    set("frame-count", data.frame_count ?? "--");
    set("event-count", data.event_count ?? 0);
    set("risk-score", data.risk_score != null ? `${Number(data.risk_score).toFixed(1)} / 100` : "--");
    set("alert-level", data.alert_level || "--");
    const faceFoundText = data.face_detected ? t("driver_v1.face_found", "Detected") : t("driver_v1.face_missing", "Face not found");
    const meshReadyText = data.face_mesh_detected ? t("driver_v1.features_ready", "Six features extracted") : t("driver_v1.features_missing", "No valid landmarks");
    set("face-state", faceFoundText);
    set("mesh-state", meshReadyText);
    setBadgeState(faceStatePill, faceFoundText, data.face_detected ? "badge-success" : "badge-warning");
    setCardState("card-face-state", data.face_detected ? "is-good" : "is-warn");
    setCardState("card-mesh-state", data.face_mesh_detected ? "is-good" : "is-warn");
    const renderFaceSignal = (valueId, cardId, active, confirmed, calibrated, confidence, defaultClear = false) => {
      let text = t("driver_v1.signal_clear", "No");
      let tone = "is-good";
      if (defaultClear) {
        text = t("driver_v1.signal_clear", "No");
      } else if (!data.face_mesh_detected) {
        text = t("driver_v1.signal_no_data", "No data");
        tone = "is-neutral";
      } else if (!calibrated && cardId !== "card-yawning-state") {
        text = `${t("driver_v1.calibrating", "Calibrating")} ${Math.round(Number(data.face_behavior_calibration_progress || 0) * 100)}%`;
        tone = "is-neutral";
      } else if (confirmed) {
        text = `${t("driver_v1.signal_confirmed", "Confirmed")} (${Number(confidence || 0).toFixed(2)})`;
        tone = "is-danger";
      } else if (active) {
        text = `${t("driver_v1.signal_confirming", "Confirming")} (${Number(confidence || 0).toFixed(2)})`;
        tone = "is-warn";
      }
      set(valueId, text);
      setCardState(cardId, tone);
    };
    renderFaceSignal(
      "yawning-state", "card-yawning-state",
      data.yawning_active, data.yawning_confirmed, true, data.yawning_confidence,
    );
    renderFaceSignal(
      "looking-away-state", "card-looking-away-state",
      data.looking_away_active, data.looking_away_confirmed,
      data.face_behavior_calibrated, data.looking_away_confidence, true,
    );
    renderFaceSignal(
      "head-down-state", "card-head-down-state",
      data.head_down_active, data.head_down_confirmed,
      data.face_behavior_calibrated, data.head_down_confidence,
    );
    const identityLabels = {
      verified: t("driver_v1.identity_verified", "Verified"),
      mismatch: t("driver_v1.identity_mismatch", "Wrong driver"),
      not_enrolled: t("driver_v1.identity_not_enrolled", "Not enrolled"),
      verifier_unavailable: t("driver_v1.identity_unavailable", "Verifier unavailable"),
      collecting: t("driver_v1.identity_collecting", "Checking"),
      identity_error: t("driver_v1.identity_error", "Verification error"),
    };
    const identityLabel = identityLabels[data.identity_status] || "--";
    const distanceText = data.identity_distance == null
      ? ""
      : ` (${Number(data.identity_distance).toFixed(3)} <= ${Number(data.identity_threshold || 0).toFixed(2)})`;
    setBadgeState(
      document.getElementById("identity-state"),
      `${identityLabel}${distanceText}`,
      data.identity_status === "verified" ? "badge-success" : (data.identity_status === "mismatch" ? "badge-warning" : "badge-neutral"),
    );
    const actionBufferReady = Number(data.driver_action_buffer_size || 0) >= Number(data.driver_action_buffer_target || 16);
    const actionState = data.driver_action_error
      ? t("driver_v1.action_error", "Action model error")
      : (data.driver_action_label
        ? `${actionText(data.driver_action_label)} (${Number(data.driver_action_confidence || 0).toFixed(2)})`
        : (data.driver_action_stable_ready
          ? `${t("driver_v1.action_low_confidence", "Low confidence")} (${Number(data.driver_action_instant_confidence || 0).toFixed(2)} < ${Number(data.driver_action_decision_threshold || 0).toFixed(2)})`
          : (actionBufferReady && data.driver_action_instant_label
          ? `${t("driver_v1.action_stabilizing", "Stabilizing")}: ${actionText(data.driver_action_instant_label)} (${Number(data.driver_action_instant_confidence || 0).toFixed(2)})`
          : t("driver_v1.action_collecting", "Collecting a 16-frame clip."))));
    set("driver-action-state", actionState);
    setCardState("card-driver-action-state", data.driver_action_event_type ? "is-warn" : (data.driver_action_label ? "is-good" : "is-neutral"));
    renderActionProbabilities(data.driver_action_probabilities);
    set("buffer-state", `${data.temporal_buffer_size ?? 0} / ${data.temporal_buffer_target ?? 15}`);
    set("action-buffer-state", `${data.driver_action_buffer_size ?? 0} / ${data.driver_action_buffer_target ?? 16}`);
    setCardState("card-buffer-state", Number(data.temporal_buffer_size || 0) > 0 ? "is-good" : "is-neutral");
    setCardState("card-action-buffer-state", Number(data.driver_action_buffer_size || 0) > 0 ? "is-good" : "is-neutral");
    set("feature-rate", data.feature_sample_fps == null ? "--" : `${Number(data.feature_sample_fps).toFixed(1)} FPS`);
    set("window-duration", data.temporal_window_seconds == null ? "--" : `${Number(data.temporal_window_seconds).toFixed(2)} s`);
    setBadgeState(modelReady, data.model_ready ? t("driver_v1.pipeline_ready", "Pipeline ready") : t("driver_v1.pipeline_missing", "Model or scaler missing"), data.model_ready ? "badge-success" : "badge-warning");
    const probability = data.drowsiness_probability;
    set("drowsiness-probability", probability == null ? "--" : `${(Number(probability) * 100).toFixed(1)}%`);
    document.getElementById("drowsiness-meter").style.width = `${Math.min(100, Math.max(0, Number(probability || 0) * 100))}%`;
    const decision = probability == null
      ? t("driver_v1.collecting", "Collecting a 15-frame sequence.")
      : (data.event_confirmed
        ? t("driver_v1.confirmed", "A drowsiness event was confirmed.")
        : (data.alert_signal
          ? t("driver_v1.pending_confirmation", "Strong signal; counting the 3-second confirmation window.")
          : (data.drowsiness_state === "drowsy"
            ? t("driver_v1.model_positive", "The model detects drowsiness, below the warning threshold.")
            : t("driver_v1.alert_prediction", "The model predicts an alert state."))));
    set("drowsiness-decision", decision);
  });
  socket.on("realtime_event", (event) => {
    const list = document.getElementById("driver-safety-events");
    const empty = list.querySelector(".muted"); if (empty) empty.remove();
    const item = document.createElement("div"); item.className = "event-card";
    const eventTitle = actionText(event.event_type) || event.event_type || t("common.event", "Event");
    item.innerHTML = `<div class="event-card-head"><strong>${eventTitle}</strong><span class="badge badge-warning">${Number(event.confidence || 0).toFixed(2)}</span></div><p>${event.explanation || ""}</p>`;
    list.prepend(item);
  });
  start.addEventListener("click", () => {
    error.classList.add("hidden");
    socket.emit("start_realtime_session", {
      behavior_model_backend: "driver_safety_v1",
      phone_detector_backend: phoneBackend,
      language: document.documentElement.lang || "vi",
    });
  });
  stop.addEventListener("click", () => { socket.emit("stop_realtime_session"); setFeed(false); });
});
