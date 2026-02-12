#!/bin/bash

# PardusDB Installation Script
# This script builds and installs PardusDB as 'pardusai' command

set -e

INSTALL_DIR="/usr/local/bin"
USER_INSTALL_DIR="$HOME/.local/bin"
BINARY_NAME="pardusai"

echo "==================================="
echo "   PardusDB Installation Script"
echo "==================================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed."
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "[1/3] Building PardusDB (release mode)..."
cargo build --release

# Check if build was successful
if [ ! -f "target/release/pardusdb" ]; then
    echo "Error: Build failed. Binary not found."
    exit 1
fi

echo "[2/3] Installing pardusai..."

# Try system-wide install first, fallback to user install
if [ -w "$INSTALL_DIR" ]; then
    cp target/release/pardusdb "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"
    FINAL_INSTALL_DIR="$INSTALL_DIR"
elif command -v sudo &> /dev/null; then
    echo "Installing to $INSTALL_DIR (sudo required)..."
    if sudo cp target/release/pardusdb "$INSTALL_DIR/$BINARY_NAME" 2>/dev/null; then
        sudo chmod +x "$INSTALL_DIR/$BINARY_NAME"
        FINAL_INSTALL_DIR="$INSTALL_DIR"
    else
        # Fallback to user directory
        echo "sudo not available, installing to user directory..."
        mkdir -p "$USER_INSTALL_DIR"
        cp target/release/pardusdb "$USER_INSTALL_DIR/$BINARY_NAME"
        chmod +x "$USER_INSTALL_DIR/$BINARY_NAME"
        FINAL_INSTALL_DIR="$USER_INSTALL_DIR"
    fi
else
    # No sudo, use user directory
    mkdir -p "$USER_INSTALL_DIR"
    cp target/release/pardusdb "$USER_INSTALL_DIR/$BINARY_NAME"
    chmod +x "$USER_INSTALL_DIR/$BINARY_NAME"
    FINAL_INSTALL_DIR="$USER_INSTALL_DIR"
fi

echo "[3/3] Verifying installation..."

# Check if the install dir is in PATH
if [[ ":$PATH:" != *":$FINAL_INSTALL_DIR:"* ]]; then
    echo ""
    echo "==================================="
    echo "   Installation Complete!"
    echo "==================================="
    echo ""
    echo "Installed to: $FINAL_INSTALL_DIR/pardusai"
    echo ""
    echo "Add this to your shell config (~/.bashrc or ~/.zshrc):"
    echo ""
    echo "    export PATH=\"\$PATH:$FINAL_INSTALL_DIR\""
    echo ""
    echo "Then run: source ~/.bashrc  (or source ~/.zshrc)"
    echo ""
    echo "After that, you can run:"
    echo "    pardusai"
    echo ""
else
    echo ""
    echo "==================================="
    echo "   Installation Successful!"
    echo "==================================="
    echo ""
    echo "You can now run PardusDB by typing:"
    echo ""
    echo "    pardusai"
    echo ""
    echo "To start with a database file:"
    echo "    pardusai mydb.pardus"
    echo ""
fi
