#!/usr/bin/env bash
# Repolect installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Bibyutatsu/Repolect/main/install.sh | bash

set -e

# ── Colours ───────────────────────────────────────────────────────────────────

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { printf "${BLUE}  ▸${NC} %s\n" "$*"; }
success() { printf "${GREEN}  ✓${NC} %s\n" "$*"; }
warn()    { printf "${YELLOW}  ⚠${NC} %s\n" "$*"; }
error()   { printf "${RED}  ✗${NC} %s\n" "$*" >&2; exit 1; }

# ── Prompt helpers ────────────────────────────────────────────────────────────

prompt_input() {
    local prompt="$1" default="$2" result=""
    if [ -n "$default" ]; then
        printf "${BOLD}  %s${NC} [${GREEN}%s${NC}]: " "$prompt" "$default" >/dev/tty
    else
        printf "${BOLD}  %s${NC}: " "$prompt" >/dev/tty
    fi
    read -r result </dev/tty
    printf "%s" "${result:-$default}"
}

prompt_secret() {
    local prompt="$1" result=""
    printf "${BOLD}  %s${NC}: " "$prompt" >/dev/tty
    read -rs result </dev/tty
    printf "\n" >/dev/tty
    printf "%s" "$result"
}

prompt_yn() {
    local prompt="$1" default="$2" result
    result=$(prompt_input "$prompt (y/n)" "$default")
    case "$result" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# ── Extras builder ────────────────────────────────────────────────────────────
# Avoids the comma-prefix/sed cleanup pattern.

EXTRAS=""
add_extra() { EXTRAS="${EXTRAS:+${EXTRAS},}$1"; }

extras_spec() {
    # Returns "repolect" or "repolect[a,b,c]"
    if [ -n "$EXTRAS" ]; then
        printf "repolect[%s]" "$EXTRAS"
    else
        printf "repolect"
    fi
}

# ── Python detection ──────────────────────────────────────────────────────────

check_python() {
    if command -v python3 &>/dev/null; then
        PY="python3"
    elif command -v python &>/dev/null; then
        PY="python"
    else
        error "Python 3 not found. Install Python 3.10+ and try again."
    fi

    PY_VERSION=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$($PY  -c 'import sys; print(sys.version_info.major)')
    PY_MINOR=$($PY  -c 'import sys; print(sys.version_info.minor)')

    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
        error "Python 3.10+ required (found $PY_VERSION)."
    fi

    success "Python $PY_VERSION"

    # Canonical user-site bin dir — works on macOS and Linux
    PY_USER_BIN=$($PY -m site --user-base 2>/dev/null || echo "$HOME/.local")
    PY_USER_BIN="${PY_USER_BIN}/bin"
}

check_pip() {
    $PY -m pip --version &>/dev/null || error "pip not found. Install pip and try again."
    success "pip found"
}

# ── PATH helpers ──────────────────────────────────────────────────────────────

# Detect the user's login shell from $SHELL, not from whatever shell is running
# this script (which may be /bin/bash when invoked via 'curl | bash' on a Zsh
# user's machine).
detect_rc_file() {
    local user_shell
    user_shell="$(basename "${SHELL:-bash}")"
    case "$user_shell" in
        zsh)  RC_FILE="$HOME/.zshrc" ;;
        fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
        *)    RC_FILE="$HOME/.bashrc" ;;
    esac
}

# Write an idempotent block to the user's RC file.
write_rc_path() {
    local bin_dir="$1"
    [ -f "$RC_FILE" ] || touch "$RC_FILE"

    if grep -q '# >>> repolect initialize >>>' "$RC_FILE" 2>/dev/null; then
        success "Shell PATH already configured in $RC_FILE"
        return
    fi

    if [ "$RC_FILE" = "$HOME/.config/fish/config.fish" ]; then
        # Fish uses a different PATH syntax
        cat >>"$RC_FILE" <<REOF

# >>> repolect initialize >>>
# !! Contents managed by Repolect installer — do not edit manually !!
fish_add_path "$bin_dir"
# <<< repolect initialize <<<
REOF
    else
        cat >>"$RC_FILE" <<REOF

# >>> repolect initialize >>>
# !! Contents managed by Repolect installer — do not edit manually !!
export PATH="${bin_dir}:\$PATH"
# <<< repolect initialize <<<
REOF
    fi
    success "PATH updated in $RC_FILE"
}

# Make binaries available in the CURRENT process — so we can call repolect
# right after install without the user needing to source anything.
activate_path() {
    local bin_dir="$1"
    case ":$PATH:" in
        *":${bin_dir}:"*) ;;   # already present
        *) export PATH="${bin_dir}:${PATH}" ;;
    esac
}

# ── pipx ──────────────────────────────────────────────────────────────────────

ensure_pipx() {
    if command -v pipx &>/dev/null; then
        success "pipx found"
        return
    fi

    info "pipx not found — installing..."
    $PY -m pip install --user pipx --quiet
    activate_path "$PY_USER_BIN"   # make pipx visible right now

    if ! command -v pipx &>/dev/null; then
        error "pipx installed but not on PATH ($PY_USER_BIN). Re-run after adding it to PATH."
    fi
    success "pipx installed"
}

# ── Ollama ────────────────────────────────────────────────────────────────────

install_ollama() {
    if command -v ollama &>/dev/null; then
        success "Ollama already installed"
        return
    fi
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed"
}

start_ollama() {
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        success "Ollama is running"
        return
    fi

    info "Starting Ollama..."
    ollama serve &>/dev/null &
    local i
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            success "Ollama is running"
            return
        fi
        sleep 1
    done
    error "Ollama failed to start. Run 'ollama serve' manually and re-run the installer."
}

pull_model() {
    local model="$1" label="$2"

    # Exact match on the name column (NAME is the first whitespace-delimited field)
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$model"; then
        success "$label model '$model' already available"
        return
    fi

    info "Pulling $label model: $model (this may take a few minutes)..."
    ollama pull "$model"
    success "$label model '$model' ready"
}

# ── Config writer ─────────────────────────────────────────────────────────────

write_config() {
    local config_dir="$HOME/.repolect"
    local config_file="$config_dir/config.yaml"
    mkdir -p "$config_dir"
    cat >"$config_file" <<EOF
# Repolect configuration — generated by install.sh

# LLM provider: "ollama" or "openai-compatible"
provider: $1
base_url: $2
model_name: $3
api_key: $4

# LLM defaults
temperature: 0.1
max_summarization_tokens: 400
max_reasoning_tokens: 1000
timeout: 60

# Embeddings (leave empty to disable semantic search)
embedding_provider: $5
embedding_model: $6
embedding_base_url: $7
EOF
    success "Config saved → $config_file"
}

# ── .gitignore ────────────────────────────────────────────────────────────────

ensure_gitignore() {
    local gitignore=".gitignore" entry=".repolect/"
    if [ ! -f "$gitignore" ]; then
        printf "%s\n" "$entry" >"$gitignore"
        success "Created $gitignore with $entry"
        return
    fi
    if grep -qxF "$entry" "$gitignore" 2>/dev/null; then
        success "$entry already in $gitignore"
        return
    fi
    printf "\n%s\n" "$entry" >>"$gitignore"
    success "Added $entry to $gitignore"
}

# ── Repolect install ──────────────────────────────────────────────────────────

install_repolect() {
    local spec
    spec="$(extras_spec)"

    # ── Try pipx (preferred: isolated env, clean upgrades) ───────────────────
    if command -v pipx &>/dev/null; then
        info "Installing $(extras_spec) via pipx..."

        # Try PyPI first; fall back to git using PEP 508 direct-reference syntax
        # so extras are correctly passed in both cases.
        if pipx install "$spec" --force 2>/dev/null || \
           pipx install "${spec} @ git+https://github.com/Bibyutatsu/Repolect.git" --force; then

            # Discover pipx's bin dir portably rather than hard-coding ~/.local/bin
            local pipx_bin
            pipx_bin="$(pipx environment --value PIPX_BIN_DIR 2>/dev/null || echo "$HOME/.local/bin")"

            activate_path "$pipx_bin"    # current session
            write_rc_path  "$pipx_bin"   # future sessions

            success "Repolect installed via pipx"
            return 0
        fi
        warn "pipx install failed — falling back to pip --user"
    fi

    # ── Fallback: pip --user ─────────────────────────────────────────────────
    info "Installing $spec via pip --user..."
    if $PY -m pip install --user --upgrade "$spec" 2>/dev/null || \
       $PY -m pip install --user --upgrade \
           "${spec} @ git+https://github.com/Bibyutatsu/Repolect.git"; then

        activate_path "$PY_USER_BIN"    # current session
        write_rc_path  "$PY_USER_BIN"   # future sessions

        success "Repolect installed via pip --user"
        return 0
    fi

    error "Failed to install Repolect. Check your Python/pip environment."
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    printf "\n${BOLD}  🧠 Repolect Installer${NC}\n"
    printf "  Vectorless code intelligence for any codebase\n\n"

    # ── Prerequisites ─────────────────────────────────────────────────────────
    check_python
    check_pip
    detect_rc_file
    printf "\n"

    # ── Step 1: LLM Provider ──────────────────────────────────────────────────
    printf "${BOLD}  Step 1: Choose your LLM provider${NC}\n"
    printf "    ${GREEN}1)${NC} Ollama             — Free, private, runs locally\n"
    printf "    ${GREEN}2)${NC} OpenAI-compatible  — OpenAI, LM Studio, Azure, etc.\n\n"
    PROVIDER_CHOICE=$(prompt_input "Select provider" "1")

    WRITE_CONFIG=true
    SHOW_CONFIG_INSTRUCTIONS=false

    case "$PROVIDER_CHOICE" in
        1|ollama)
            PROVIDER="ollama"
            add_extra "ollama"   # ensures the ollama Python SDK extra is installed

            printf "\n"
            install_ollama
            start_ollama

            printf "\n"
            info "Browse models at: https://ollama.com/search"

            # Show already-installed models so the user can reuse one
            INSTALLED=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' | tr '\n' '  ')
            if [ -n "$INSTALLED" ]; then
                info "Already installed: ${INSTALLED}"
            fi

            LLM_MODEL=$(prompt_input "LLM model name" "qwen3.5:4b")
            pull_model "$LLM_MODEL" "LLM"

            BASE_URL="http://localhost:11434"
            API_KEY=""

            printf "\n"
            if prompt_yn "Configure embeddings for semantic search?" "y"; then
                EMBED_MODEL=$(prompt_input "Embedding model name" "qwen3-embedding:0.6b")
                pull_model "$EMBED_MODEL" "Embedding"
                EMBED_PROVIDER="ollama"
                EMBED_BASE_URL=""
            else
                warn "Skipping embeddings. Configure later in ~/.repolect/config.yaml"
                EMBED_MODEL="" EMBED_PROVIDER="" EMBED_BASE_URL=""
            fi
            ;;

        2|openai*)
            PROVIDER="openai-compatible"

            printf "\n${BOLD}  How would you like to configure your provider?${NC}\n"
            printf "    ${GREEN}1)${NC} Enter config via CLI now\n"
            printf "    ${GREEN}2)${NC} I'll create the config file myself\n\n"
            CONFIG_CHOICE=$(prompt_input "Select" "1")

            case "$CONFIG_CHOICE" in
                1)
                    printf "\n"
                    BASE_URL=$(prompt_input "Base URL" "https://api.openai.com/v1")
                    LLM_MODEL=$(prompt_input "Model name" "gpt-4o-mini")
                    API_KEY=$(prompt_secret "API key (hidden)")
                    [ -z "$API_KEY" ] && warn "No API key provided. Set it later in ~/.repolect/config.yaml"

                    printf "\n"
                    if prompt_yn "Configure embeddings for semantic search?" "y"; then
                        EMBED_PROVIDER="openai-compatible"
                        EMBED_BASE_URL=$(prompt_input "Embedding base URL" "$BASE_URL")
                        EMBED_MODEL=$(prompt_input "Embedding model" "text-embedding-3-small")
                    else
                        warn "Skipping embeddings. Configure later in ~/.repolect/config.yaml"
                        EMBED_PROVIDER="" EMBED_MODEL="" EMBED_BASE_URL=""
                    fi
                    ;;
                2)
                    WRITE_CONFIG=false
                    SHOW_CONFIG_INSTRUCTIONS=true
                    LLM_MODEL="" BASE_URL="" API_KEY=""
                    EMBED_PROVIDER="" EMBED_MODEL="" EMBED_BASE_URL=""
                    ;;
                *)
                    error "Invalid choice: $CONFIG_CHOICE"
                    ;;
            esac
            ;;

        *)
            error "Invalid choice: $PROVIDER_CHOICE"
            ;;
    esac

    # ── Step 2: Optional extras ───────────────────────────────────────────────
    printf "\n${BOLD}  Step 2: Optional extras${NC}\n\n"

    prompt_yn "Install FalkorDB graph backend? (recommended)" "y" && add_extra "graph"
    prompt_yn "Install visualization (Streamlit graph explorer)?" "n"  && add_extra "viz"

    # ── Step 3: Install ───────────────────────────────────────────────────────
    printf "\n${BOLD}  Step 3: Installing $(extras_spec)${NC}\n\n"
    ensure_pipx
    install_repolect

    # ── Step 4: Gitignore ─────────────────────────────────────────────────────
    ensure_gitignore

    # ── Step 5: Write config ──────────────────────────────────────────────────
    if [ "$WRITE_CONFIG" = true ]; then
        printf "\n"
        write_config \
            "$PROVIDER" "$BASE_URL" "$LLM_MODEL" "$API_KEY" \
            "$EMBED_PROVIDER" "$EMBED_MODEL" "$EMBED_BASE_URL"
    fi

    # ── Done ──────────────────────────────────────────────────────────────────
    printf "\n${GREEN}${BOLD}  ✅ Repolect is ready!${NC}\n\n"

    printf "  ${YELLOW}${BOLD}⚡ To activate in this terminal:${NC}\n"
    printf "     source %s\n\n" "$RC_FILE"

    printf "  ${BOLD}Quick start:${NC}\n"
    printf "     cd your-project/\n"
    printf "     repolect analyze\n"
    printf "     repolect ask \"how does authentication work?\"\n"

    echo "$EXTRAS" | grep -q "viz" && \
        printf "     repolect viz          # launch graph explorer\n"

    printf "\n  ${BOLD}To add extras later:${NC}\n"
    printf "     pipx install --force 'repolect[viz]'\n"
    printf "     pipx install --force 'repolect[graph,viz]'\n"

    printf "\n  ${BOLD}Config:${NC}  ~/.repolect/config.yaml\n"
    printf "  ${BOLD}Docs:${NC}    https://github.com/Bibyutatsu/Repolect\n"

    if [ "$SHOW_CONFIG_INSTRUCTIONS" = true ]; then
        printf "\n${YELLOW}${BOLD}  ⚠  Manual config — run these commands:${NC}\n\n"
        printf "     mkdir -p ~/.repolect\n"
        printf "     cat > ~/.repolect/config.yaml << 'EOF'\n"
        printf "     provider: openai-compatible\n"
        printf "     base_url: https://api.openai.com/v1\n"
        printf "     model_name: gpt-4o-mini\n"
        printf "     api_key: YOUR_API_KEY\n"
        printf "     temperature: 0.1\n"
        printf "     max_summarization_tokens: 400\n"
        printf "     max_reasoning_tokens: 1000\n"
        printf "     timeout: 60\n"
        printf "     embedding_provider: openai-compatible\n"
        printf "     embedding_model: text-embedding-3-small\n"
        printf "     embedding_base_url: https://api.openai.com/v1\n"
        printf "     EOF\n"
    fi

    printf "\n"
}

main "$@"