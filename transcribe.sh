#!/usr/bin/env bash

set -euo pipefail

CALL_ID=""
AUDIO_PATH=""

usage() {
  echo "Usage:"
  echo "  $0 --call-id <CALL_ID> [--audio-path <LOCAL_AUDIO_PATH>]"
  echo "  OR"
  echo "  $0 --audio-path <LOCAL_AUDIO_PATH>"
  echo
  echo "  --call-id      Optional ID used to build the GCS object path: gs://speaker-audio/<CALL_ID>.wav"
  echo "                 If provided, audio will be downloaded from GCS."
  echo "  --audio-path   Local path for the audio file."
  echo "                 If --call-id is given, this is the download destination."
  echo "                 If --call-id is omitted, this is the already-existing local audio file."
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --call-id)
      CALL_ID="${2-}"
      shift 2
      ;;
    --audio-path)
      AUDIO_PATH="${2-}"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      echo "Unexpected positional argument: $1"
      usage
      ;;
  esac
done

# Require at least one of CALL_ID or AUDIO_PATH
if [[ -z "$CALL_ID" && -z "$AUDIO_PATH" ]]; then
  echo "Error: at least one of --call-id or --audio-path is required."
  usage
fi

# If CALL_ID is provided, we need AUDIO_PATH as the download destination
if [[ -n "$CALL_ID" && -z "$AUDIO_PATH" ]]; then
  AUDIO_PATH="/tmp/${CALL_ID}.wav"
  #echo "Error: when using --call-id, you must also provide --audio-path as the download destination."
  #usage
fi

if [[ -n "$CALL_ID" ]]; then
  # Download from GCS to the given AUDIO_PATH
  gcloud storage cp "gs://speaker-audio/${CALL_ID}.wav" "$AUDIO_PATH"
else
  # No CALL_ID: assume AUDIO_PATH is an existing local file and do nothing here.
  echo "Using existing local audio file at: $AUDIO_PATH"
fi

# Start the tunnel if not running already (checking port 9002)
if ! lsof -i:9002 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Starting SSH tunnel on port 9002..."
  gcloud compute ssh ubuntu@knowl-stt-inference --tunnel-through-iap --zone=asia-south1-b -- -L 0.0.0.0:9002:10.160.0.14:8080 -N -n &
else
  echo "SSH tunnel already running on port 9002."
fi

/home/bhups/silero-vad/bin/python example/example_client_usage.py "$AUDIO_PATH" true
