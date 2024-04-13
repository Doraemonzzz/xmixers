#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

sh "${ABSOLUTE_PATH}"/requirements.txt