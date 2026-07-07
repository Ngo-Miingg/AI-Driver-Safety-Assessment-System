(function () {
  function t(key, fallback) {
    return (window.I18N && window.I18N[key]) || fallback || key;
  }

  var statusEl = document.getElementById("socket-status");
  if (!statusEl || typeof io === "undefined") {
    return;
  }

  var socket = io();

  socket.on("connect", function () {
    statusEl.textContent = t("home.connected", "Connected");
    socket.emit("ping_server", { source: "home" });
  });

  socket.on("disconnect", function () {
    statusEl.textContent = t("js.disconnected", "Disconnected");
  });

  socket.on("server_status", function (message) {
    if (message && message.status) {
      statusEl.textContent = message.status === "ok" ? t("home.connected", "Connected") : message.status;
    }
  });
})();
