# Source me: headless software OpenGL (Mesa llvmpipe through EGL) for MuJoCo rendering on
# machines without a usable EGL driver (no display, no NVIDIA EGL vendor library).
#
#   source tools/softgl.sh
#   CUDA_VISIBLE_DEVICES= .venv/bin/unisupermimic-video --backend mujoco --ckpt ...
#
# The first call installs Mesa from conda-forge into tools/.softgl with a private micromamba
# (about 300 MB, no root needed); later calls only export the environment.
_softgl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.softgl"
if [ ! -f "$_softgl_dir/env/share/glvnd/egl_vendor.d/50_mesa.json" ]; then
  echo "softgl: installing Mesa into $_softgl_dir" >&2
  mkdir -p "$_softgl_dir"
  if [ ! -x "$_softgl_dir/bin/micromamba" ]; then
    # python's tarfile instead of tar -j: bzip2 is often missing on minimal images
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$_softgl_dir/mm.tar.bz2" \
      && python3 -c 'import sys, tarfile; tarfile.open(sys.argv[1]).extract("bin/micromamba", sys.argv[2])' \
        "$_softgl_dir/mm.tar.bz2" "$_softgl_dir" \
      && rm "$_softgl_dir/mm.tar.bz2" || return 1 2>/dev/null || exit 1
  fi
  "$_softgl_dir/bin/micromamba" create -y -q -r "$_softgl_dir/root" -p "$_softgl_dir/env" \
    -c conda-forge mesalib libopengl libgl || return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="$_softgl_dir/env/lib:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="$_softgl_dir/env/share/glvnd/egl_vendor.d"
export LIBGL_DRIVERS_PATH="$_softgl_dir/env/lib/dri" GALLIUM_DRIVER=llvmpipe
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
unset _softgl_dir
