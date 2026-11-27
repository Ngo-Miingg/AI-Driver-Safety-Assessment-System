document.addEventListener("DOMContentLoaded", () => {
  const socket = window.io ? window.io() : null;
  const startButton = document.getElementById("start-realtime");
  const stopButton = document.getElementById("stop-realtime");
  const statusBadge = document.getElementById("realtime-status");
  const statusHelper = document.getElementById("realtime-status-helper");
  const connectionBadge = document.getElementById("realtime-connection");
  const sessionLink = document.getElementById("realtime-session-link");
  const eventList = document.getElementById("realtime-event-list");
  const snapshotImage = document.getElementById("realtime-snapshot-image");
  const snapshotEmpty = document.getElementById("realtime-snapshot-empty");
  const errorBox = document.getElementById("realtime-error");
  const logBox = document.getElementById("realtime-log");
  const streamImage = document.querySelector(".realtime-stream");
  const streamPlaceholder = document.getElementById("realtime-stream-placeholder");
  const skeletonPreviewImage = document.getElementById("skeleton-preview-image");
  const skeletonEmpty = document.getElementById("skeleton-empty");
  const audioLevelBar = document.getElementById("audio-level-bar");
  const audioActivityBadge = document.getElementById("metric-audio-activity");
  const audioDetail = document.getElementById("metric-audio-detail");
  const skeletonQuality = document.getElementById("metric-skeleton-quality");

  const fields = {
    frameCount: document.getElementById("metric-frame-count"),
    riskScore: document.getElementById("metric-risk-score"),
    safetyScore: document.getElementById("metric-safety-score"),
    alertLevel: document.getElementById("metric-alert-level"),
    faceState: document.getElementById("metric-face-state"),
    videoState: document.getElementById("metric-video-state"),
    audioState: document.getElementById("metric-audio-state"),
    skeletonState: document.getElementById("metric-skeleton-state"),
    activeLabels: document.getElementById("metric-active-labels"),
    temporalBufferSizeSide: document.getElementById("metric-temporal-buffer-size-side"),
    warning: document.getElementById("metric-warning"),
    eventCount: document.getElementById("metric-event-count"),
    analysisMode: document.getElementById("metric-analysis-mode"),
    liveSession: document.getElementById("metric-live-session"),
    liveModeNote: document.getElementById("metric-live-mode-note"),
  };

  const statusIndicators = {
    faceDetected: {
      card: document.getElementById("indicator-face-detected"),
      value: document.getElementById("metric-face-detected-status"),
    },
    eyeClosed: {
      card: document.getElementById("indicator-eye-closed"),
      value: document.getElementById("metric-eye-closed-status"),
    },
    yawn: {
      card: document.getElementById("indicator-yawn"),
      value: document.getElementById("metric-yawn-status"),
    },
    lookingAway: {
      card: document.getElementById("indicator-looking-away"),
      value: document.getElementById("metric-looking-away-status"),
    },
    headDown: {
      card: document.getElementById("indicator-head-down"),
      value: document.getElementById("metric-head-down-status"),
    },
    phoneDetected: {
      card: document.getElementById("indicator-phone-detected"),
      value: document.getElementById("metric-phone-detected-status"),
    },
    drowsiness: {
      card: document.getElementById("indicator-drowsiness"),
      value: document.getElementById("metric-drowsiness-status"),
    },
  };

  const VI_FALLBACK = {
    running: "Dang chay",
    idle: "Dang cho",
    connected: "Da ket noi",
    disconnected: "Mat ket noi",
    monitoring: "Dang theo doi",
    video_live: "Video truc tiep",
    audio_waiting: "Dang cho audio tu mic",
    skeleton_waiting: "Dang cho skeleton",
    multimodal_live_note: "Model multimodal dang nhan video, audio va skeleton that.",
    audio_idle: "Mic dang yen",
    audio_active: "Mic dang co tin hieu",
    skeleton_ready: "Skeleton san sang",
    skeleton_missing: "Chua thay tu the",
    audio_model_hint: "Muc am thanh tu mic dang duoc dua vao model cung video va skeleton.",
  };

  let running = false;
  let lastSessionId = null;
  let streamUrl = streamImage ? (streamImage.dataset.feedUrl || streamImage.getAttribute("src") || "") : "";
  let audioContext = null;
  let audioStream = null;
  let audioSource = null;
  let audioProcessor = null;
  let audioGain = null;
  let audioChunkSize = 4096;
  let localAudioRms = 0;
  const micSampleRate = { value: 0 };

  function getCurrentLanguage() {
    return (window.APP_LANGUAGE || document.documentElement.lang || "en").toLowerCase();
  }

  function t(key, fallback) {
    return (window.I18N && window.I18N[key]) || fallback || key;
  }

  function normalizeKey(value) {
    return String(value || "").trim().toLowerCase().replace(/\s+/g, "_");
  }

  function translateValue(value, fallback) {
    const normalized = normalizeKey(value);
    if (!normalized) return fallback || "--";
    const translated = window.I18N ? window.I18N[`value.${normalized}`] : null;
    if (translated) return translated;
    if (getCurrentLanguage() === "vi" && VI_FALLBACK[normalized]) return VI_FALLBACK[normalized];
    return fallback || normalized.replace(/_/g, " ");
  }

  function setText(id, value) {
    const node = fields[id];
    if (!node) return;
    node.textContent = value ?? "--";
  }

  function setIndicator(indicator, active, activeText, inactiveText, options = {}) {
    if (!indicator || !indicator.value || !indicator.card) return;
    const positiveWhenActive = options.positiveWhenActive === true;
    if (active == null) {
      indicator.value.textContent = options.emptyText || t("realtime.no_data", "No data");
      indicator.card.classList.remove("is-good", "is-risk", "is-unknown");
      indicator.card.classList.add("is-unknown");
      return;
    }
    indicator.value.textContent = active ? activeText : inactiveText;
    indicator.card.classList.remove("is-good", "is-risk", "is-unknown");
    const activeClass = positiveWhenActive ? "is-good" : "is-risk";
    const inactiveClass = positiveWhenActive ? "is-risk" : "is-good";
    indicator.card.classList.add(active ? activeClass : inactiveClass);
  }

  function log(message, tone = "neutral") {
    if (!logBox) return;
    const item = document.createElement("div");
    item.className = `timeline-log ${tone}`;
    item.textContent = message;
    logBox.prepend(item);
  }

  function setState(nextRunning) {
    running = nextRunning;
    if (startButton) startButton.disabled = nextRunning;
    if (stopButton) stopButton.disabled = !nextRunning;
    if (statusBadge) {
      statusBadge.textContent = nextRunning ? t("js.running", VI_FALLBACK.running) : t("realtime.idle", VI_FALLBACK.idle);
      statusBadge.dataset.state = nextRunning ? "running" : "idle";
    }
    if (statusHelper) {
      statusHelper.textContent = nextRunning
        ? t("js.status_helper", "Realtime monitoring status: ") + t("js.running", VI_FALLBACK.running)
        : t("js.waiting_update", "Waiting for the first realtime update.");
    }
  }

  function resetLiveReadouts() {
    [
      "frameCount",
      "riskScore",
      "safetyScore",
      "alertLevel",
      "faceState",
      "videoState",
      "audioState",
      "skeletonState",
      "activeLabels",
      "temporalBufferSizeSide",
      "warning",
      "eventCount",
      "analysisMode",
      "liveSession",
      "liveModeNote",
    ].forEach((id) => setText(id, "--"));
    Object.values(statusIndicators).forEach((indicator) => {
      setIndicator(indicator, null, "", "");
    });
    if (eventList) eventList.innerHTML = "";
    if (logBox) logBox.innerHTML = "";
    if (snapshotImage) {
      snapshotImage.removeAttribute("src");
      snapshotImage.classList.remove("is-visible");
    }
    if (snapshotEmpty) snapshotEmpty.classList.remove("hidden");
    clearSkeleton();
    setAudioMeter(0);
  }

  function detachPreviewStream() {
    if (!streamImage) return;
    if (!streamUrl) streamUrl = streamImage.dataset.feedUrl || streamImage.getAttribute("src") || "";
    streamImage.removeAttribute("src");
    if (streamPlaceholder) streamPlaceholder.classList.remove("hidden");
  }

  function restorePreviewStream() {
    if (!streamImage || !streamUrl) return;
    streamImage.setAttribute("src", `${streamUrl}${streamUrl.includes("?") ? "&" : "?"}ts=${Date.now()}`);
    if (streamPlaceholder) streamPlaceholder.classList.add("hidden");
  }

  function setConnection(isConnected) {
    if (!connectionBadge) return;
    connectionBadge.textContent = isConnected ? t("js.connected_short", VI_FALLBACK.connected) : t("js.disconnected", VI_FALLBACK.disconnected);
    connectionBadge.className = `badge ${isConnected ? "badge-success" : "badge-warning"}`;
  }

  function updateSnapshot(imageUrl) {
    if (!snapshotImage || !snapshotEmpty) return;
    snapshotImage.src = imageUrl;
    snapshotImage.classList.add("is-visible");
    snapshotEmpty.classList.add("hidden");
  }

  function setAudioMeter(rms) {
    localAudioRms = Math.max(0, Math.min(1, rms || 0));
    if (audioLevelBar) {
      audioLevelBar.style.transform = `scaleX(${Math.max(0.02, localAudioRms)})`;
    }
    if (audioActivityBadge) {
      const active = localAudioRms >= 0.035;
      audioActivityBadge.textContent = active ? t("realtime.audio_active", VI_FALLBACK.audio_active) : t("realtime.audio_idle", VI_FALLBACK.audio_idle);
      audioActivityBadge.className = `badge ${active ? "badge-success" : "badge-neutral"}`;
    }
    if (audioDetail) {
      audioDetail.textContent = `${t("realtime.audio_model_hint", VI_FALLBACK.audio_model_hint)} RMS: ${localAudioRms.toFixed(3)}`;
    }
  }

  function clearSkeleton() {
    if (skeletonPreviewImage) {
      skeletonPreviewImage.removeAttribute("src");
      skeletonPreviewImage.classList.remove("is-visible");
    }
    if (skeletonEmpty) skeletonEmpty.classList.remove("hidden");
    if (skeletonQuality) {
      skeletonQuality.textContent = t("realtime.skeleton_missing", VI_FALLBACK.skeleton_missing);
      skeletonQuality.className = "badge badge-neutral";
    }
  }

  function updateSkeleton(imageUrl, pointCount) {
    if (!imageUrl) {
      clearSkeleton();
      return;
    }
    if (skeletonPreviewImage) {
      skeletonPreviewImage.src = imageUrl;
      skeletonPreviewImage.classList.add("is-visible");
    }
    if (skeletonEmpty) skeletonEmpty.classList.add("hidden");
    if (skeletonQuality) {
      skeletonQuality.textContent = `${t("realtime.skeleton_ready", VI_FALLBACK.skeleton_ready)} (${pointCount || 0})`;
      skeletonQuality.className = "badge badge-success";
    }
  }

  function addEventItem(event) {
    if (!eventList) return;
    const explanation = event.explanation_key ? t(event.explanation_key, event.explanation || "") : (event.explanation || "");
    const item = document.createElement("div");
    item.className = "event-card";
    item.innerHTML = `
      <div class="event-card-head">
        <strong>${translateValue(event.event_type, event.event_type || "event")}</strong>
        <span class="badge ${event.severity === "critical" ? "badge-danger" : (event.severity === "high" ? "badge-warning" : "badge-info")}">${translateValue(event.severity, event.severity || "info")}</span>
      </div>
      <div class="muted">${t("common.start", "Start")}: ${Number(event.start_time_seconds || 0).toFixed(2)}s | ${t("common.end", "End")}: ${Number(event.end_time_seconds || 0).toFixed(2)}s</div>
      <div class="muted">${t("common.confidence", "Confidence")}: ${Number(event.confidence || 0).toFixed(2)}</div>
      <p>${explanation}</p>
    `;
    eventList.prepend(item);
  }

  function updateMetrics(data) {
    const isConfirmedAction = Boolean(data.should_log_event) || normalizeKey(data.action_label) === "normal_driving";
    const displayAction = isConfirmedAction
      ? translateValue(
          data.action_label || data.predicted_label || data.face_state || data.active_labels,
          data.action_label || data.predicted_label || data.active_labels || "--",
        )
      : t("realtime.monitoring", VI_FALLBACK.monitoring);

    setText("frameCount", data.frame_count != null ? data.frame_count : "--");
    setText("riskScore", data.risk_score != null ? Number(data.risk_score).toFixed(1) : "--");
    setText("safetyScore", data.safety_score != null ? Number(data.safety_score).toFixed(1) : "--");
    setText("alertLevel", translateValue(data.alert_level, data.alert_level || "--"));
    setText("faceState", displayAction);
    setText("videoState", data.video_state || t("realtime.video_live", VI_FALLBACK.video_live));
    setText("audioState", data.audio_state || t("realtime.audio_waiting", VI_FALLBACK.audio_waiting));
    setText("skeletonState", data.skeleton_state || t("realtime.skeleton_waiting", VI_FALLBACK.skeleton_waiting));
    setText("activeLabels", isConfirmedAction ? (data.active_labels || t("realtime.monitoring", VI_FALLBACK.monitoring)) : t("realtime.monitoring", VI_FALLBACK.monitoring));
    setText("temporalBufferSizeSide", data.temporal_buffer_size != null ? data.temporal_buffer_size : "--");
    setText("warning", data.warning_key ? t(data.warning_key, data.warning || "--") : translateValue(data.warning, data.warning || "--"));
    setText("eventCount", data.event_count != null ? data.event_count : "--");
    setText("analysisMode", translateValue(data.analysis_mode, data.analysis_mode || "--"));
    setText("liveSession", data.session_id != null ? `#${data.session_id}` : "--");
    setText("liveModeNote", data.live_mode_note || t("realtime.multimodal_live_note", VI_FALLBACK.multimodal_live_note));

    setIndicator(
      statusIndicators.faceDetected,
      data.face_detected != null ? Boolean(data.face_detected) : null,
      t("js.detected", "Detected"),
      t("js.missing", "Missing"),
      { positiveWhenActive: true },
    );
    setIndicator(statusIndicators.eyeClosed, data.eye_closed, t("common.yes", "Yes"), t("common.no", "No"));
    setIndicator(statusIndicators.yawn, data.yawn_candidate, t("common.yes", "Yes"), t("common.no", "No"));
    setIndicator(statusIndicators.lookingAway, data.looking_away, t("common.yes", "Yes"), t("common.no", "No"));
    setIndicator(statusIndicators.headDown, data.head_down, t("common.yes", "Yes"), t("common.no", "No"));
    setIndicator(
      statusIndicators.phoneDetected,
      data.phone_detected != null ? data.phone_detected : (data.phone_detected_frame_count != null ? Number(data.phone_detected_frame_count || 0) > 0 : null),
      t("common.yes", "Yes"),
      t("common.no", "No"),
    );
    if (data.analysis_mode === "driver_safety_v1" || data.drowsiness_probability != null) {
      const probability = data.drowsiness_probability;
      const state = data.event_confirmed
        ? t("realtime.drowsiness_confirmed", "Confirmed")
        : data.alert_signal
          ? t("realtime.drowsiness_warning", "Warning")
          : data.drowsiness_state === "alert"
            ? t("realtime.drowsiness_clear", "Alert")
            : data.drowsiness_state === "collecting_features"
              ? t("realtime.drowsiness_collecting", "Collecting")
              : t("realtime.drowsiness_unknown", "No data");
      const text = probability == null ? state : `${state} (${(Number(probability) * 100).toFixed(1)}%)`;
      setIndicator(
        statusIndicators.drowsiness,
        probability == null ? null : Boolean(data.alert_signal || data.event_confirmed),
        text,
        text,
        { emptyText: t("realtime.drowsiness_collecting", "Collecting") },
      );
    }

    if (statusHelper) {
      statusHelper.textContent = `${t("js.status_helper", "Realtime monitoring status: ")}${translateValue(data.alert_level, data.alert_level || "--")} | ${t("realtime.warning", "Warning")}: ${data.warning_key ? t(data.warning_key, data.warning || "--") : (data.warning || "--")}`;
    }

    updateSkeleton(data.skeleton_image, Array.isArray(data.pose_points) ? data.pose_points.length : 0);

    if (data.session_id) {
      lastSessionId = data.session_id;
    }
    if (sessionLink && lastSessionId) {
      sessionLink.href = `/sessions/${lastSessionId}`;
      sessionLink.classList.remove("hidden");
    }
  }

  function readPayload() {
    const behavior = document.getElementById("behavior-model-select");
    const behaviorBackend = behavior ? behavior.value : "mmrt_multimodal_full_v2_t4x2";
    return {
      behavior_model_backend: behaviorBackend,
      phone_detector_backend: "disabled",
      language: getCurrentLanguage(),
    };
  }

  async function startAudioCapture() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Browser does not support microphone capture.");
    }
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioContextCtor();
    micSampleRate.value = audioContext.sampleRate;
    audioSource = audioContext.createMediaStreamSource(audioStream);
    audioProcessor = audioContext.createScriptProcessor(audioChunkSize, 1, 1);
    audioGain = audioContext.createGain();
    audioGain.gain.value = 0.0;

    audioProcessor.onaudioprocess = (event) => {
      if (!running || !socket) return;
      const input = event.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(input);
      let sum = 0;
      for (let i = 0; i < chunk.length; i += 1) {
        sum += chunk[i] * chunk[i];
      }
      setAudioMeter(Math.sqrt(sum / Math.max(chunk.length, 1)));
      socket.emit("realtime_audio_chunk", {
        samples: chunk.buffer,
        sample_rate: micSampleRate.value || audioContext.sampleRate || 16000,
      });
    };

    audioSource.connect(audioProcessor);
    audioProcessor.connect(audioGain);
    audioGain.connect(audioContext.destination);
  }

  async function stopAudioCapture() {
    setAudioMeter(0);
    if (audioProcessor) {
      try { audioProcessor.disconnect(); } catch (err) {}
      audioProcessor.onaudioprocess = null;
      audioProcessor = null;
    }
    if (audioSource) {
      try { audioSource.disconnect(); } catch (err) {}
      audioSource = null;
    }
    if (audioGain) {
      try { audioGain.disconnect(); } catch (err) {}
      audioGain = null;
    }
    if (audioContext) {
      try { await audioContext.close(); } catch (err) {}
      audioContext = null;
    }
    if (audioStream) {
      audioStream.getTracks().forEach((track) => track.stop());
      audioStream = null;
    }
  }

  async function beginMonitoring() {
    if (!socket) {
      throw new Error("Socket.IO client is unavailable.");
    }
    resetLiveReadouts();
    detachPreviewStream();
    await startAudioCapture();
    socket.emit("start_realtime_session", readPayload());
  }

  function stopMonitoring() {
    if (socket) socket.emit("stop_realtime_session");
    stopAudioCapture();
    detachPreviewStream();
  }

  if (!socket) {
    log("Socket.IO client is unavailable.", "error");
    if (errorBox) {
      errorBox.textContent = "Socket.IO client is unavailable.";
      errorBox.classList.remove("hidden");
    }
    return;
  }

  socket.on("connect", () => {
    setConnection(true);
    socket.emit("set_realtime_language", { language: getCurrentLanguage() });
    log(t("js.connected_realtime", "Connected to realtime server."));
  });

  socket.on("disconnect", () => {
    setConnection(false);
    log(t("js.disconnected_realtime", "Disconnected from realtime server."), "warning");
  });

  socket.on("server_status", (payload) => {
    if (payload?.status === "connected") {
      log(t("js.server_handshake", "Server handshake completed."));
    } else if (payload?.status === "ok") {
      log(`${t("js.server_ok", "Server reply: ")}${payload.echo ?? "ok"}`);
    }
  });

  socket.on("realtime_status", (payload) => {
    if (!payload) return;
    if (payload.status === "starting") {
      setState(true);
      if (statusHelper) statusHelper.textContent = t("js.realtime_starting", "Realtime session is starting...");
      log(t("js.realtime_starting", "Realtime session is starting..."));
    } else if (payload.status === "running") {
      setState(true);
      restorePreviewStream();
      if (statusHelper) statusHelper.textContent = `${t("js.realtime_running", "Session #")}${payload.session_id}${t("js.realtime_running_suffix", " is running on camera ")}${payload.camera_index}.`;
      log(`${t("js.realtime_running", "Session #")}${payload.session_id}${t("js.realtime_running_suffix", " is running on camera ")}${payload.camera_index}.`);
      if (payload.session_id) {
        lastSessionId = payload.session_id;
        if (sessionLink) {
          sessionLink.href = `/sessions/${payload.session_id}`;
          sessionLink.classList.remove("hidden");
        }
      }
      updateMetrics(payload);
    } else if (payload.status === "stopping") {
      if (statusHelper) statusHelper.textContent = t("js.realtime_stopping", "Realtime session is stopping...");
      log(t("js.realtime_stopping", "Realtime session is stopping..."));
    } else if (payload.status === "completed") {
      setState(false);
      if (statusHelper) statusHelper.textContent = `${t("js.realtime_completed", "Realtime session completed. Safety ")}${payload.final_safety_score}${t("js.realtime_completed_middle", ", risk ")}${payload.final_risk_score}.`;
      log(`${t("js.realtime_completed", "Realtime session completed. Safety ")}${payload.final_safety_score}${t("js.realtime_completed_middle", ", risk ")}${payload.final_risk_score}.`);
      stopAudioCapture();
      detachPreviewStream();
    }
  });

  socket.on("realtime_update", (payload) => {
    updateMetrics(payload || {});
  });

  socket.on("realtime_snapshot", (payload) => {
    if (payload?.image) updateSnapshot(payload.image);
  });

  socket.on("realtime_event", (payload) => {
    if (!payload) return;
    addEventItem(payload);
    log(`${t("common.event", "Event")}: ${translateValue(payload.event_type, payload.event_type)}`, payload.severity === "critical" ? "error" : "warning");
  });

  socket.on("realtime_completed", (payload) => {
    if (!payload) return;
    setState(false);
    if (statusHelper) statusHelper.textContent = `${t("js.realtime_saved", "Saved session #")}${payload.session_id}.`;
    if (payload.session_url && sessionLink) {
      sessionLink.href = payload.session_url;
      sessionLink.classList.remove("hidden");
    }
    stopAudioCapture();
    detachPreviewStream();
    log(`${t("js.realtime_saved", "Saved session #")}${payload.session_id}.`, "success");
  });

  socket.on("realtime_error", (payload) => {
    const message = payload?.message || t("js.realtime_error", "Realtime error.");
    if (errorBox) {
      errorBox.textContent = message;
      errorBox.classList.remove("hidden");
    }
    if (statusHelper) statusHelper.textContent = message;
    setState(false);
    stopAudioCapture();
    detachPreviewStream();
    log(message, "error");
  });

  if (startButton) {
    startButton.addEventListener("click", async () => {
      if (errorBox) errorBox.classList.add("hidden");
      try {
        await beginMonitoring();
      } catch (err) {
        const message = err?.message || "Unable to start realtime audio capture.";
        if (errorBox) {
          errorBox.textContent = message;
          errorBox.classList.remove("hidden");
        }
        log(message, "error");
      }
    });
  }

  if (stopButton) {
    stopButton.addEventListener("click", () => {
      stopMonitoring();
    });
  }

  clearSkeleton();
  setAudioMeter(0);
  setState(false);
  setConnection(false);
});
