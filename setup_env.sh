#!/bin/bash
# Creates the conda environments HEDeST needs.
#
#   ./setup_env.sh                  # both environments
#   ./setup_env.sh --hedest-only    # only hedest-env (training, features, analysis)
#   ./setup_env.sh --hovernet-only  # only hovernet-env (mask, segmentation)
#
# Environment names can be overridden: HEDEST_ENV=my-env ./setup_env.sh
# Existing environments are left alone unless --force is given.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEDEST_ENV="${HEDEST_ENV:-hedest-env}"
HOVERNET_ENV="${HOVERNET_ENV:-hovernet-env}"
PYTHONPATH_VALUE="$REPO:$REPO/external:$REPO/external/hovernet:$REPO/external/mocov3"

DO_HEDEST=1
DO_HOVERNET=1
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --hedest-only) DO_HOVERNET=0 ;;
        --hovernet-only) DO_HEDEST=0 ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install miniconda or anaconda first." >&2
    exit 1
fi

# The classic solver takes several minutes on conda-forge; use libmamba when it is there.
SOLVER_ARGS=()
CONDA_BASE="$(conda info --base)"
if "$CONDA_BASE/bin/python" -c "import conda_libmamba_solver" >/dev/null 2>&1 \
   && conda create --help 2>&1 | grep -q -- "--solver"; then
    SOLVER_ARGS=(--solver=libmamba)
    echo "==> Using the libmamba solver"
fi

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

create_env() {
    local name="$1" pyversion="$2" requirements="$3"

    if env_exists "$name"; then
        if [ "$FORCE" -eq 1 ]; then
            echo "==> Removing the existing environment $name"
            conda env remove -y -n "$name" >/dev/null
        else
            echo "==> $name already exists, skipping (use --force to rebuild it)."
            return 0
        fi
    fi

    # openslide is a system library, so it comes from conda rather than pip. It is
    # installed together with python to keep it to a single solve.
    echo "==> Creating $name (python $pyversion, openslide)"
    conda create -y -n "$name" "${SOLVER_ARGS[@]+"${SOLVER_ARGS[@]}"}" -c conda-forge "python=$pyversion" "openslide>=3.4.1"

    echo "==> Installing the python requirements in $name"
    conda run --no-capture-output -n "$name" python -m pip install --upgrade pip
    conda run --no-capture-output -n "$name" python -m pip install -r "$REPO/$requirements"

    # the repository is used as a source tree, and openslide needs the conda libraries
    local prefix
    prefix="$(conda run -n "$name" python -c 'import sys; print(sys.prefix)')"
    conda env config vars set -n "$name" \
        PYTHONPATH="$PYTHONPATH_VALUE" \
        LD_LIBRARY_PATH="$prefix/lib" >/dev/null

    echo "==> $name is ready"
}

check_env() {
    local name="$1"
    shift
    echo "==> Checking $name"
    conda run --no-capture-output -n "$name" python -c "$@"
}

if [ "$DO_HEDEST" -eq 1 ]; then
    create_env "$HEDEST_ENV" 3.9 requirements.txt
    check_env "$HEDEST_ENV" "
import torch, timm, openslide, torch_scatter
import hedest.pipeline, hedest.slide, hedest.main
from hedest.features.hoptimus import extract_hoptimus_embeddings
print(f'  torch {torch.__version__} | timm {timm.__version__} | cuda available: {torch.cuda.is_available()}')
print('  hedest imports fine')
"
fi

if [ "$DO_HOVERNET" -eq 1 ]; then
    create_env "$HOVERNET_ENV" 3.8 requirements-hovernet.txt
    check_env "$HOVERNET_ENV" "
import torch, cv2, openslide, docopt, imgaug
print(f'  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')
print('  hovernet dependencies fine')
"
fi

cat <<EOF

Done.
  conda activate $HEDEST_ENV     # pipeline, features, training, analysis
  conda activate $HOVERNET_ENV   # mask and segmentation stages

In the pipeline configuration, point the HoVer-Net stages at the second one:
  mask:
    python: \$(conda run -n $HOVERNET_ENV which python)
  segmentation:
    python: \$(conda run -n $HOVERNET_ENV which python)
EOF
