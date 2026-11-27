(function () {
  const video = document.getElementById("face-enrollment-video");
  const canvas = document.getElementById("face-enrollment-canvas");
  const placeholder = document.getElementById("face-enrollment-placeholder");
  const startButton = document.getElementById("face-camera-start");
  const captureButton = document.getElementById("face-camera-capture");
  const stopButton = document.getElementById("face-camera-stop");
  const stepText = document.getElementById("face-enrollment-step");
  const countText = document.getElementById("face-enrollment-count");
  const enrollmentForm = video ? video.closest("form") : null;
  const thumbs = document.getElementById("face-enrollment-thumbs");
  const wrap = video ? video.closest(".face-enrollment-video-wrap") : null;

  if (!video || !canvas || !startButton || !captureButton || !stopButton || !enrollmentForm) {
    return;
  }

  const vi = (window.APP_LANGUAGE || "en") === "vi";
  const labels = vi
    ? {
        idle: "Camera đăng ký đang chờ",
        opening: "Đang mở camera...",
        ready: "Camera sẵn sàng. Giữ mặt trong khung.",
        denied: "Không mở được camera. Hãy kiểm tra quyền camera hoặc dùng ảnh upload.",
        capturing: "Đang lấy mẫu",
        done: "Đã lấy đủ mẫu. Bấm Đăng ký khuôn mặt để lưu.",
        count: "Số mẫu đã chụp",
        poses: [
          "Nhìn thẳng vào camera",
          "Quay mặt nhẹ sang trái",
          "Quay mặt nhẹ sang phải",
          "Ngẩng mặt nhẹ",
          "Cúi mặt nhẹ",
          "Nhìn thẳng lại, giữ yên",
        ],
      }
    : {
        idle: "Camera enrollment is idle",
        opening: "Opening camera...",
        ready: "Camera is ready. Keep your face inside the frame.",
        denied: "Could not open camera. Check camera permission or use file upload.",
        capturing: "Capturing sample",
        done: "Samples captured. Press Enroll face to save.",
        count: "Captured samples",
        poses: [
          "Look straight at the camera",
          "Turn slightly left",
          "Turn slightly right",
          "Tilt slightly up",
          "Tilt slightly down",
          "Look straight again and hold still",
        ],
      };

  const MAX_CAPTURE_WIDTH = 640;
  const MAX_CAPTURE_HEIGHT = 360;
  const JPEG_QUALITY = 0.58;

  let stream = null;
  let captured = [];

  function setStatus(text) {
    if (stepText) stepText.textContent = text;
  }

  function updateCount() {
    if (countText) countText.textContent = `${labels.count}: ${captured.length}`;
  }

  function renderThumb(dataUrl) {
    if (!thumbs) return;
    const img = document.createElement("img");
    img.src = dataUrl;
    img.alt = `${labels.count} ${captured.length}`;
    thumbs.appendChild(img);
  }

  async function openCamera() {
    setStatus(labels.opening);
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 540 },
          facingMode: "user",
        },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      if (wrap) wrap.classList.add("has-stream");
      if (placeholder) placeholder.classList.add("hidden");
      startButton.disabled = true;
      captureButton.disabled = false;
      stopButton.disabled = false;
      setStatus(labels.ready);
    } catch (error) {
      setStatus(labels.denied);
      startButton.disabled = false;
      captureButton.disabled = true;
      stopButton.disabled = true;
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    video.srcObject = null;
    if (wrap) wrap.classList.remove("has-stream");
    if (placeholder) placeholder.classList.remove("hidden");
    startButton.disabled = false;
    captureButton.disabled = true;
    stopButton.disabled = true;
    setStatus(labels.idle);
  }

  function captureFrame(poseHint) {
    const sourceWidth = video.videoWidth || 960;
    const sourceHeight = video.videoHeight || 540;
    const scale = Math.min(MAX_CAPTURE_WIDTH / sourceWidth, MAX_CAPTURE_HEIGHT / sourceHeight, 1);
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    captured.push({ data_url: dataUrl, pose_hint: poseHint });
    renderThumb(dataUrl);
    updateCount();
  }

  async function captureGuidedSamples() {
    if (!stream) {
      await openCamera();
    }
    if (!stream) return;
    captured = [];
    if (thumbs) thumbs.innerHTML = "";
    updateCount();
    captureButton.disabled = true;

    for (let index = 0; index < labels.poses.length; index += 1) {
      const pose = labels.poses[index];
      setStatus(`${labels.capturing} ${index + 1}/${labels.poses.length}: ${pose}`);
      await new Promise((resolve) => setTimeout(resolve, index === 0 ? 700 : 1100));
      captureFrame(pose);
    }

    setStatus(labels.done);
    captureButton.disabled = false;
  }

  async function dataUrlToFile(dataUrl, filename) {
    const response = await fetch(dataUrl);
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type || "image/jpeg" });
  }

  async function submitWithCameraSamples(event) {
    if (!captured.length) {
      return;
    }
    event.preventDefault();
    startButton.disabled = true;
    captureButton.disabled = true;
    stopButton.disabled = true;
    setStatus(vi ? "Đang gửi mẫu khuôn mặt..." : "Submitting face samples...");
    const formData = new FormData(enrollmentForm);
    formData.delete("face_samples_json");
    for (let index = 0; index < captured.length; index += 1) {
      const sample = captured[index];
      const file = await dataUrlToFile(sample.data_url, `camera_sample_${String(index + 1).padStart(2, "0")}.jpg`);
      formData.append("face_samples", file, file.name);
    }
    const response = await fetch(enrollmentForm.action, {
      method: "POST",
      body: formData,
      credentials: "same-origin",
    });
    if (response.redirected) {
      window.location.assign(response.url);
      return;
    }
    if (response.ok) {
      window.location.assign(response.url || enrollmentForm.action);
      return;
    }
    setStatus(vi ? "Gửi mẫu thất bại." : "Failed to submit samples.");
    startButton.disabled = false;
    captureButton.disabled = false;
    stopButton.disabled = false;
  }

  startButton.addEventListener("click", openCamera);
  captureButton.addEventListener("click", captureGuidedSamples);
  stopButton.addEventListener("click", stopCamera);
  enrollmentForm.addEventListener("submit", submitWithCameraSamples);
  window.addEventListener("beforeunload", stopCamera);
  setStatus(labels.idle);
  updateCount();
})();
