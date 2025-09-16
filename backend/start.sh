#!/usr/bin/env bash
set -e

# start celery in background (logs go to stdout), CONCURRENCY set to 1
celery -A main.celery_app worker --loglevel=info -c 1 &

# start your web server in foreground (Render expects a process listening on $PORT)
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
