#!/bin/bash
set -e

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — add your HF_TOKEN and/or Azure keys before testing."
fi

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl
pip3 install --no-cache-dir requests

echo "Dev container ready. API: http://localhost:9000  docs: http://localhost:9000/docs"
