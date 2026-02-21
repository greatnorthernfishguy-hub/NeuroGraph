#!/usr/bin/env bash
#
# deploy.sh — One-command NeuroGraph deployment to OpenClaw
#
# Usage:
#   ./deploy.sh              # Full deployment
#   ./deploy.sh --deps-only  # Only install dependencies
#   ./deploy.sh --files-only # Only deploy files (skip dependencies)
#   ./deploy.sh --uninstall  # Remove deployed files
#
# Environment:
#   NEUROGRAPH_HOME           — Override data directory (default: ~/.neurograph)
#   NEUROGRAPH_SKILL_DIR      — Override module install dir (default: $NEUROGRAPH_HOME/modules)
#
# Config file: ~/.neurograph.conf  (written on first deploy, read by all components)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Unified path resolution — same order as neurograph_paths.py:
#   1. NEUROGRAPH_HOME env var
#   2. ~/.neurograph.conf JSON file
#   3. NEUROGRAPH_WORKSPACE_DIR env var (legacy)
#   4. Default: ~/.neurograph
_resolve_home() {
    # 1. Explicit env var
    if [ -n "${NEUROGRAPH_HOME:-}" ]; then
        echo "$NEUROGRAPH_HOME"
        return
    fi
    # 2. Config file
    local conf="$HOME/.neurograph.conf"
    if [ -f "$conf" ]; then
        local from_conf
        from_conf="$(python3 -c "import json; print(json.load(open('$conf')).get('neurograph_home',''))" 2>/dev/null || true)"
        if [ -n "$from_conf" ]; then
            echo "$from_conf"
            return
        fi
    fi
    # 3. Legacy env var
    if [ -n "${NEUROGRAPH_WORKSPACE_DIR:-}" ]; then
        echo "$NEUROGRAPH_WORKSPACE_DIR"
        return
    fi
    # 4. Default
    echo "$HOME/.neurograph"
}

NEUROGRAPH_HOME="$(_resolve_home)"
SKILL_DIR="${NEUROGRAPH_SKILL_DIR:-$NEUROGRAPH_HOME/modules}"
WORKSPACE_DIR="$NEUROGRAPH_HOME"
CHECKPOINT_DIR="$WORKSPACE_DIR/checkpoints"
BIN_DIR="$HOME/.local/bin"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*" >&2; }

# ------------------------------------------------------------------
# Dependency installation
# ------------------------------------------------------------------
install_deps() {
    info "Installing dependencies..."

    # Core deps (always needed)
    pip3 install --break-system-packages --no-cache-dir \
        "numpy>=1.24.0" "scipy>=1.10.0" "msgpack>=1.0.0" 2>/dev/null || \
    pip3 install --no-cache-dir \
        "numpy>=1.24.0" "scipy>=1.10.0" "msgpack>=1.0.0"

    # sentence-transformers from source (bleeding edge, no broken deps)
    info "Installing sentence-transformers from source..."
    pip3 install --break-system-packages --no-deps --no-cache-dir \
        "sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git" 2>/dev/null || \
    pip3 install --no-deps --no-cache-dir \
        "sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git" || \
    warn "sentence-transformers install failed — hash fallback will be used"

    # transformers (gets modern huggingface_hub)
    pip3 install --break-system-packages --no-cache-dir transformers tqdm 2>/dev/null || \
    pip3 install --no-cache-dir transformers tqdm || \
    warn "transformers install failed — hash fallback will be used"

    # CPU-only PyTorch
    pip3 install --break-system-packages --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip3 install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu || \
    warn "PyTorch install failed — hash fallback will be used"

    # Optional: beautifulsoup4, PyPDF2
    pip3 install --break-system-packages --no-cache-dir \
        "beautifulsoup4>=4.12.0" "PyPDF2>=3.0.0" 2>/dev/null || \
    pip3 install --no-cache-dir \
        "beautifulsoup4>=4.12.0" "PyPDF2>=3.0.0" || \
    warn "Optional deps (beautifulsoup4, PyPDF2) failed — URL/PDF extraction unavailable"

    info "Dependencies installed"
}

# ------------------------------------------------------------------
# File deployment
# ------------------------------------------------------------------
deploy_files() {
    info "Deploying NeuroGraph files..."

    # Write unified config so all components agree on the data path
    local conf="$HOME/.neurograph.conf"
    if [ ! -f "$conf" ]; then
        echo "{\"neurograph_home\": \"$NEUROGRAPH_HOME\"}" > "$conf"
        info "Created $conf -> $NEUROGRAPH_HOME"
    else
        info "Config exists: $conf (using configured path)"
    fi

    # Create directories
    mkdir -p "$SKILL_DIR"
    mkdir -p "$SKILL_DIR/scripts"
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$BIN_DIR"

    # Core engine files
    cp "$SCRIPT_DIR/neuro_foundation.py" "$SKILL_DIR/"
    cp "$SCRIPT_DIR/universal_ingestor.py" "$SKILL_DIR/"

    # Integration files
    cp "$SCRIPT_DIR/openclaw_hook.py" "$SKILL_DIR/"
    cp "$SCRIPT_DIR/openclaw_hook.py" "$SKILL_DIR/scripts/"
    cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/"

    # Migration framework
    cp "$SCRIPT_DIR/neurograph_migrate.py" "$SKILL_DIR/"

    # CLI tool
    cp "$SCRIPT_DIR/feed-syl" "$BIN_DIR/feed-syl"
    chmod +x "$BIN_DIR/feed-syl"

    # Also copy feed-syl to home for convenience
    cp "$SCRIPT_DIR/feed-syl" "$HOME/feed-syl"
    chmod +x "$HOME/feed-syl"

    # Ensure ~/.local/bin is on PATH
    if ! echo "$PATH" | grep -q "$BIN_DIR"; then
        if [ -f "$HOME/.bashrc" ]; then
            if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
                info "Added $BIN_DIR to PATH in ~/.bashrc"
            fi
        fi
    fi

    # Configure OpenClaw if config exists
    OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
    if [ -f "$OPENCLAW_CONFIG" ]; then
        # Check if neurograph skill is already configured
        if python3 -c "
import json, sys
with open('$OPENCLAW_CONFIG') as f:
    cfg = json.load(f)
skills = cfg.get('skills', {}).get('entries', {})
if 'neurograph' not in skills:
    skills['neurograph'] = {
        'enabled': True,
        'env': {'NEUROGRAPH_HOME': '$NEUROGRAPH_HOME'}
    }
    cfg.setdefault('skills', {})['entries'] = skills
    with open('$OPENCLAW_CONFIG', 'w') as f:
        json.dump(cfg, f, indent=2)
    print('configured')
else:
    print('already_configured')
" 2>/dev/null; then
            info "OpenClaw configuration updated"
        fi
    else
        # Create minimal OpenClaw config
        mkdir -p "$HOME/.openclaw"
        cat > "$OPENCLAW_CONFIG" << 'JSONEOF'
{
  "skills": {
    "entries": {
      "neurograph": {
        "enabled": true,
        "env": {
          "NEUROGRAPH_HOME": "~/.neurograph"
        }
      }
    }
  }
}
JSONEOF
        info "Created OpenClaw configuration"
    fi

    info "Files deployed to $SKILL_DIR"
    info "CLI tool installed at $BIN_DIR/feed-syl"
}

# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------
verify() {
    info "Verifying installation..."
    local ok=true

    # Check core files
    for f in neuro_foundation.py universal_ingestor.py openclaw_hook.py SKILL.md; do
        if [ ! -f "$SKILL_DIR/$f" ]; then
            error "Missing: $SKILL_DIR/$f"
            ok=false
        fi
    done

    # Check CLI
    if [ ! -x "$BIN_DIR/feed-syl" ]; then
        error "Missing or not executable: $BIN_DIR/feed-syl"
        ok=false
    fi

    # Check Python imports
    if python3 -c "
import sys
sys.path.insert(0, '$SKILL_DIR')
from neuro_foundation import Graph
from universal_ingestor import UniversalIngestor, SimpleVectorDB
from openclaw_hook import NeuroGraphMemory
print('imports_ok')
" 2>/dev/null | grep -q "imports_ok"; then
        info "Python imports OK"
    else
        error "Python import check failed"
        ok=false
    fi

    # Check embedding backend
    if python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
e = model.encode(['test'])
print(f'embeddings_ok dim={e.shape[1]}')
" 2>/dev/null; then
        info "Real embeddings available"
    else
        warn "sentence-transformers unavailable — using hash fallback"
    fi

    if $ok; then
        info "Verification passed"
    else
        error "Verification failed — check errors above"
        return 1
    fi
}

# ------------------------------------------------------------------
# Uninstall
# ------------------------------------------------------------------
uninstall() {
    warn "Removing deployed NeuroGraph files..."
    rm -f "$BIN_DIR/feed-syl"
    rm -f "$HOME/feed-syl"
    rm -rf "$SKILL_DIR"
    info "Files removed (checkpoints preserved in $CHECKPOINT_DIR)"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
case "${1:-}" in
    --deps-only)
        install_deps
        ;;
    --files-only)
        deploy_files
        verify
        ;;
    --uninstall)
        uninstall
        ;;
    --verify)
        verify
        ;;
    *)
        echo "========================================"
        echo " NeuroGraph Deployment"
        echo "========================================"
        echo ""
        install_deps
        echo ""
        deploy_files
        echo ""
        verify
        echo ""
        info "Deployment complete!"
        echo ""
        echo "Quick start:"
        echo "  feed-syl --status              # Check status"
        echo "  feed-syl --text 'Hello world'  # Ingest text"
        echo "  feed-syl --workspace           # Ingest OpenClaw docs"
        echo ""
        ;;
esac
