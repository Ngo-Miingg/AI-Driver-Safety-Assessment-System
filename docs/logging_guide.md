# Logging Guide

## Log Location

Default log directory:

```text
logs/
```

Main app log:

```text
logs/app.log
```

The log directory can be changed with:

```text
LOG_DIR=D:\Work\HocKy3\ComputerVision\MDDAD\logs
```

## Log Level

Configure with:

```text
LOG_LEVEL=INFO
```

Recommended values:

- `INFO` for normal demo and local operation.
- `DEBUG` when investigating realtime SocketIO or camera behavior.
- `WARNING` for quieter production-like runs.

## Events Logged

- Application startup configuration summary.
- SocketIO connect/disconnect.
- Realtime start/stop requests.
- Camera open success/failure.
- First realtime frame/update/snapshot.
- Realtime errors and finalization.

## Troubleshooting

If realtime appears frozen, check:

1. Browser DevTools console.
2. Flask terminal output.
3. `logs/app.log`.
4. `/debug/camera` in debug mode.
