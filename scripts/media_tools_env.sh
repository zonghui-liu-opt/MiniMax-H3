#!/usr/bin/env bash
# Optional shared setup; source this from the existing SGLang environment.
h3_configure_media_tools() {
  [[ -n "${H3_MEDIA_TOOLS_DIR:-}" ]] || return 0

  local tools_dir shim_dir tool
  if ! tools_dir="$(cd -- "$H3_MEDIA_TOOLS_DIR" 2>/dev/null && pwd -P)"; then
    printf 'H3_MEDIA_TOOLS_DIR 目录不可访问: %s\n' "$H3_MEDIA_TOOLS_DIR" >&2
    return 2
  fi
  for tool in ffmpeg ffprobe; do
    if [[ ! -f "$tools_dir/$tool" || ! -x "$tools_dir/$tool" ]]; then
      printf 'H3_MEDIA_TOOLS_DIR 中缺少可执行工具: %s\n' "$tools_dir/$tool" >&2
      return 2
    fi
  done

  shim_dir="$(mktemp -d "${TMPDIR:-/tmp}/minimax-h3-media.XXXXXXXX")" || return 2
  for tool in ffmpeg ffprobe; do
    # Execute the absolute target, including when it is itself a wrapper script.
    # Never add the other environment's python/sglang or shared libraries.
    if ! { printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$tools_dir/$tool" > "$shim_dir/$tool" &&
           chmod +x "$shim_dir/$tool"; }; then
      rm -f -- "$shim_dir/ffmpeg" "$shim_dir/ffprobe"
      rmdir -- "$shim_dir"
      return 2
    fi
  done
  # Keep this directory alive across exec and for all service worker processes.
  export PATH="$shim_dir:$PATH"
  printf 'H3 media tools: ffmpeg=%s; ffprobe=%s\n' "$tools_dir/ffmpeg" "$tools_dir/ffprobe" >&2
}

h3_configure_media_tools
