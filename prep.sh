#!/bin/sh

confirm() {
  while true; do
    printf "$1? (y/N) "
    read -r input
    case "$input" in
    (y | Y) return 0 ;;
    (n | N | "") return 1 ;;
    (*) echo "Please type y or n" ;;
    esac
  done
}

echo "Linting..."
cargo clippy -- -D warnings

if confirm "Attempt to apply lints (which will modify files)"; then
  cargo clippy --allow-dirty --allow-staged --fix -- -D warnings
fi

echo "Checking formatting..."
cargo fmt --all --check

if confirm "Attempt to apply formatting (which will modify files)"; then
  cargo fmt --all
fi

echo "Checking for errors..."
cargo check

echo "Running tests..."
cargo test
