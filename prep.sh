#!/usr/bin/env sh

printf "Linting...\n";
cargo clippy -- -D warnings;
while :; do
	printf "Attempt to apply lints? This will write changes to files. (y/N): "
	read -r input;
	input="$(printf '%s' "${input}" | tr '[:upper:]' '[:lower:]')";
	case "${input}" in
		("y" | "yes")
			printf "Applying lints...\n";
			cargo clippy --allow-dirty --allow-staged --fix -- -D warnings;
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
cargo fmt --all --check;
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
