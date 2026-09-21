#!/usr/bin/env bash
set -euo pipefail

# Record the current VNC/X11 desktop, including the Gazebo GUI.
if [[ -z "${DISPLAY:-}" ]]; then
  echo 'DISPLAY is unset; run this in the Jetson VNC desktop session.' >&2
  exit 1
fi
if ! command -v xwininfo >/dev/null 2>&1; then
  echo 'Missing xwininfo' >&2
  exit 1
fi
size="$(xwininfo -root | awk '/Width:/{w=$2} /Height:/{h=$2} END{print w "x" h}')"
if [[ ! "$size" =~ ^[0-9]+x[0-9]+$ ]]; then
  echo 'Could not read the VNC desktop size.' >&2
  exit 1
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/work/gazebo-screen-$(date -u +%Y%m%dT%H%M%SZ).mp4}"
mkdir -p "$(dirname "$output")"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "Recording ${size} from ${DISPLAY} to ${output}; press q to finish."
  exec ffmpeg -y -f x11grab -framerate 10 -video_size "$size" -i "${DISPLAY}+0,0" \
    -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p "$output"
fi
for plugin in ximagesrc videoconvert videoscale x264enc h264parse mp4mux; do
  if ! gst-inspect-1.0 "$plugin" >/dev/null 2>&1; then
    echo "Missing GStreamer plugin: $plugin" >&2
    exit 1
  fi
done
width="${size%x*}"
height="${size#*x}"
record_height=$((height * 1280 / width / 2 * 2))
echo "Recording ${size} from ${DISPLAY} to ${output}; press Ctrl+C to finish."
exec gst-launch-1.0 -e -q \
  ximagesrc display-name="$DISPLAY" use-damage=0 ! video/x-raw,framerate=8/1 ! \
  videoconvert ! videoscale ! video/x-raw,width=1280,height="$record_height",format=I420 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=2500 ! h264parse ! \
  mp4mux ! filesink location="$output"
