#!/usr/bin/env bash
set -e

# start celery in background if enabled (logs go to stdout), CONCURRENCY set to 1
if [ "${ENABLE_CELERY:-true}" != "false" ] && [ "${ENABLE_REDIS:-true}" != "false" ]; then
    echo "Starting Celery worker..."
    celery -A main.celery_app worker --loglevel=info -c 1 &
else
    echo "Celery/Redis disabled by environment flag; running in local thread mode."
fi

# start your web server in foreground (Render expects a process listening on $PORT)
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}

