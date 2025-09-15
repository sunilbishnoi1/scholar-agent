#!/bin/bash

# Run the web server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app