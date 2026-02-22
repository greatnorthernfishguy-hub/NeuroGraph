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
#   NEUROGRAPH_WORKSPACE_DIR  — Override workspace (default: ~/.openclaw/neurograph)
#   NEUROGRAPH_SKILL_DIR      — Override skill dir (default: ~/.openclaw/skills/neurograph)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="${NEUROGRAPH_SKILL_DIR:-$HOME/.openclaw/skills/neurograph}"
WORKSPACE_DIR="${NEUROGRAPH_WORKSPACE_DIR:-$HOME/.openclaw/neurograph}"
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

    # CPU-only PyTorch (install BEFORE sentence-transformers so it picks up
    # the local torch backend instead of trying inference providers)
    pip3 install --break-system-packages --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip3 install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu || \
    warn "PyTorch install failed — hash fallback will be used"

    # sentence-transformers from PyPI (pinned stable release).
    # NOTE: Previous versions installed from git HEAD which pulled v5.x+
    # that added inference provider backends (OpenAI, Google, Voyage) and
    # emitted API key warnings even when only using local torch embeddings.
    # Pinning to stable PyPI release avoids this issue.
    info "Installing sentence-transformers (stable release)..."
    pip3 install --break-system-packages --no-cache-dir \
        "sentence-transformers>=3.0.0,<6.0.0" 2>/dev/null || \
    pip3 install --no-cache-dir \
        "sentence-transformers>=3.0.0,<6.0.0" || \
    warn "sentence-transformers install failed — hash fallback will be used"

    # transformers + tqdm (pinned to avoid v5 inference provider issues)
    pip3 install --break-system-packages --no-cache-dir \
        "transformers>=4.41.0" tqdm 2>/dev/null || \
    pip3 install --no-cache-dir \
        "transformers>=4.41.0" tqdm || \
    warn "transformers install failed — hash fallback will be used"

    # Optional: beautifulsoup4, PyPDF2
    pip3 install --break-system-packages --no-cache-dir \
        "beautifulsoup4>=4.12.0" "PyPDF2>=3.0.0" 2>/dev/null || \
    pip3 install --no-cache-dir \
        "beautifulsoup4>=4.12.0" "PyPDF2>=3.0.0" || \
    warn "Optional deps (beautifulsoup4, PyPDF2) failed — URL/PDF extraction unavailable"

    # watchdog for GUI file watcher
    pip3 install --break-system-packages --no-cache-dir "watchdog>=3.0.0" 2>/dev/null || \
    pip3 install --no-cache-dir "watchdog>=3.0.0" || \
    warn "watchdog install failed — GUI file watcher will use polling fallback"

    info "Dependencies installed"
}

# ------------------------------------------------------------------
# File deployment
# ------------------------------------------------------------------
deploy_files() {
    info "Deploying NeuroGraph files..."

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

    # NG-Lite ecosystem files
    cp "$SCRIPT_DIR/ng_lite.py" "$SKILL_DIR/"
    cp "$SCRIPT_DIR/ng_bridge.py" "$SKILL_DIR/"
    cp "$SCRIPT_DIR/ng_peer_bridge.py" "$SKILL_DIR/"

    # ET Module Manager
    mkdir -p "$SKILL_DIR/et_modules"
    cp "$SCRIPT_DIR/et_modules/__init__.py" "$SKILL_DIR/et_modules/"
    cp "$SCRIPT_DIR/et_modules/manager.py" "$SKILL_DIR/et_modules/"
    cp "$SCRIPT_DIR/et_module.json" "$SKILL_DIR/"

    # Migration framework
    cp "$SCRIPT_DIR/neurograph_migrate.py" "$SKILL_DIR/"

    # CES (Cognitive Enhancement Suite) files
    for cesfile in ces_config.py stream_parser.py activation_persistence.py surfacing.py ces_monitoring.py; do
        if [ -f "$SCRIPT_DIR/$cesfile" ]; then
            cp "$SCRIPT_DIR/$cesfile" "$SKILL_DIR/"
        fi
    done

    # Management GUI
    if [ -f "$SCRIPT_DIR/neurograph_gui.py" ]; then
        cp "$SCRIPT_DIR/neurograph_gui.py" "$SKILL_DIR/"
        info "GUI deployed to $SKILL_DIR/neurograph_gui.py"

        # Desktop entry (Linux application launcher)
        DESKTOP_DIR="$HOME/.local/share/applications"
        mkdir -p "$DESKTOP_DIR"
        cat > "$DESKTOP_DIR/neurograph.desktop" << DESKTOPEOF
[Desktop Entry]
Type=Application
Name=NeuroGraph Manager
GenericName=Cognitive Architecture Manager
Comment=Manage NeuroGraph updates, ingestion, and monitoring
Exec=python3 $SKILL_DIR/neurograph_gui.py
Icon=preferences-system
Categories=Utility;Development;Science;
Terminal=false
StartupWMClass=neurograph
Keywords=neurograph;ai;memory;ingestion;snn;
DESKTOPEOF
        chmod +x "$DESKTOP_DIR/neurograph.desktop"
        info "Desktop entry installed at $DESKTOP_DIR/neurograph.desktop"

        # GUI data directories
        mkdir -p "$HOME/.neurograph/inbox"
        mkdir -p "$HOME/.neurograph/ingested"
        mkdir -p "$HOME/.neurograph/logs"
        info "GUI directories created at ~/.neurograph/"
    fi

    # CLI tools
    cp "$SCRIPT_DIR/feed-syl" "$BIN_DIR/feed-syl"
    chmod +x "$BIN_DIR/feed-syl"
    cp "$SCRIPT_DIR/neurograph" "$BIN_DIR/neurograph"
    chmod +x "$BIN_DIR/neurograph"

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

    # Suppress HuggingFace inference provider API key warnings.
    # NeuroGraph uses LOCAL torch-based embeddings only — no OpenAI, Google,
    # or Voyage API keys are needed.  These env vars prevent noisy warnings
    # from sentence-transformers v5+ / transformers v5+ inference providers.
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q 'TRANSFORMERS_VERBOSITY' "$HOME/.bashrc"; then
            cat >> "$HOME/.bashrc" << 'ENVEOF'

# NeuroGraph: suppress HuggingFace inference provider warnings
# (local torch embeddings only — TID controls all external API calls)
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
export TOKENIZERS_PARALLELISM=false
ENVEOF
            info "Added HuggingFace warning suppression to ~/.bashrc"
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
        'env': {'NEUROGRAPH_WORKSPACE_DIR': '$WORKSPACE_DIR'}
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
          "NEUROGRAPH_WORKSPACE_DIR": "~/.openclaw/neurograph"
        }
      }
    }
  }
}
JSONEOF
        info "Created OpenClaw configuration"
    fi

    # --- ET Module Manager registration ---
    ET_MODULES_DIR="${ET_MODULES_DIR:-$HOME/.et_modules}"
    mkdir -p "$ET_MODULES_DIR/shared_learning"

    python3 -c "
import json, time

registry_path = '$ET_MODULES_DIR/registry.json'
try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {'modules': {}}

registry['modules']['neurograph'] = {
    'module_id': 'neurograph',
    'display_name': 'NeuroGraph Foundation',
    'version': '0.6.0',
    'install_path': '$SKILL_DIR',
    'git_remote': 'https://github.com/greatnorthernfishguy-hub/NeuroGraph.git',
    'git_branch': 'main',
    'entry_point': 'openclaw_hook.py',
    'ng_lite_version': '1.0.0',
    'dependencies': [],
    'service_name': '',
    'api_port': 0,
    'registered_at': time.time(),
}
registry['last_updated'] = time.time()

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)

print('registered')
" 2>/dev/null && info "Registered with ET Module Manager at $ET_MODULES_DIR" || \
    warn "ET Module Manager registration skipped (non-critical)"

    info "Files deployed to $SKILL_DIR"
    info "CLI tools installed at $BIN_DIR/feed-syl and $BIN_DIR/neurograph"
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

    # Check CLIs
    for cli in feed-syl neurograph; do
        if [ ! -x "$BIN_DIR/$cli" ]; then
            error "Missing or not executable: $BIN_DIR/$cli"
            ok=false
        fi
    done

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

    # Check GUI
    if [ -f "$SKILL_DIR/neurograph_gui.py" ]; then
        info "GUI: installed"
    else
        warn "GUI: not found (optional)"
    fi

    # Check embedding backend — suppress provider warnings (NeuroGraph uses
    # local torch only; TID controls all external API calls)
    if python3 -c "
import os, warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2', backend='torch')
except TypeError:
    model = SentenceTransformer('all-MiniLM-L6-v2')
e = model.encode(['test'])
print(f'embeddings_ok dim={e.shape[1]} backend=torch')
" 2>/dev/null; then
        info "Real embeddings available (local torch — no external API keys needed)"
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
    rm -f "$BIN_DIR/neurograph"
    rm -f "$HOME/feed-syl"
    rm -rf "$SKILL_DIR"
    rm -f "$HOME/.local/share/applications/neurograph.desktop"
    info "Files removed (checkpoints preserved in $CHECKPOINT_DIR)"
    info "Note: ~/.neurograph/ (inbox/ingested data) preserved"
    info "Note: ~/.et_modules/ (shared learning data) preserved"
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
        echo "  neurograph status              # System health check"
        echo "  neurograph verify              # Verify installation"
        echo "  feed-syl --status              # Check status"
        echo "  feed-syl --text 'Hello world'  # Ingest text"
        echo "  feed-syl --workspace           # Ingest OpenClaw docs"
        echo ""
        ;;
esac
