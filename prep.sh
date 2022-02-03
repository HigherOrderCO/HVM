#!/usr/bin/env sh

if test -n "$(git diff Cargo.lock)"; then
	if test -z "$(command -v crate2nix)"; then
		printf "\033[33mWARNING:\033[0m \`crate2nix\` doesn't appear to be installed.\n";
		printf "Please install it with \`nix-env -i -f https://github.com/kolloch/crate2nix/tarball/0.10.0\`,\n";
		printf "or run this script in the Nix development shell with \`nix develop\`.\n";
	else
		printf 'Updating "Cargo.nix"...\n';
		crate2nix generate;
	fi;
fi;
printf "Linting...\n";
cargo clippy -- -D warnings;
while :; do
	printf "Attempt to apply lints? This will write changes to files. (y/N): "
	read -r input;
	input="$(printf '%s' "${input}" | tr '[:upper:]' '[:lower:]')";
	case "${input}" in
		("y" | "yes")
			printf "Applying lints...\n";
			cargo clippy -- -D warnings --fix;
			break;
		;;
		("" | "n" | "no")
			break;
		;;
		(*)
			printf "\033[33mWARNING:\033[0m Unrecognised input \"%b\". Ignoring.\n" "${input}";
		;;
	esac;
done;
printf "Checking formatting...\n";
cargo fmt --all -- --check;
while :; do
	printf "Attempt to apply formatting? This will write changes to files. (y/N): "
	read -r input;
	input="$(printf '%s' "${input}" | tr '[:upper:]' '[:lower:]')";
	case "${input}" in
		("y" | "yes")
			printf "Applying formatting...\n";
			cargo fmt --all;
			break;
		;;
		("" | "n" | "no")
			break;
		;;
		(*)
			printf "\033[33mWARNING:\033[0m Unrecognised input \"%b\". Ignoring.\n" "${input}";
		;;
	esac;
done;
printf "Checking for errors...\n";
cargo check;
printf "Running tests...\n";
cargo test;
