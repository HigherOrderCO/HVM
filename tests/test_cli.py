#!/usr/bin/env python3

import os
from pathlib import Path
import argparse
import subprocess

from typing import Iterator
from dataclasses import dataclass
import difflib
import json

# TODO: port cases from /bench/
# TODO: CI
# TODO: run cases in parallel?

@dataclass
class TestResult:
    test_name: str
    case_name: str
    ok: bool


def run_test(
    folder_path: Path, bin_path: Path, differ: difflib.Differ
) -> Iterator[TestResult]:
    test_name = folder_path.name
    folder_path = folder_path.absolute()

    print(f"Testing: {test_name}")

    code_path = folder_path.joinpath(f"{test_name}.hvm")
    assert (
        code_path.exists()
    ), f"{test_name} test case must have a hvm source code file!"

    specs_path = folder_path.joinpath(f"{test_name}.json")
    assert specs_path.exists(), f"{test_name} test case must have an I/O file!"

    with open(specs_path.absolute(), "r") as jf:
        specs = json.load(jf)
    for case_name in specs:
        spec = specs[case_name]
        print(f"Testing file '{case_name}'... ".ljust(28), end="")

        case_args = spec["input"]
        expected_out = spec["output"]
        cmd = [bin_path, "run", code_path.absolute(), case_args]

        p = subprocess.run(cmd, capture_output=True)
        test_out = p.stdout.decode("utf-8").strip()

        diff = differ.compare(
            test_out.splitlines(keepends=True),
            expected_out.splitlines(keepends=True),
        )
        diff_lines = [line for line in diff if not line.startswith("  ")]
        if not diff_lines:
            print("✅ PASS")
            yield TestResult(test_name, case_name, True)
        else:
            print("❌ FAILED")
            for line in diff_lines:
                print(line)
            yield TestResult(test_name, case_name, False)


def run_tests() -> Iterator[TestResult]:
    differ = difflib.Differ()

    tests_folder = Path(__file__).parent.resolve()

    bin_path = tests_folder.parent.joinpath("target", "debug", "hvm")
    assert bin_path.is_file(), "HVM must be already compiled"

    for entry in tests_folder.iterdir():
        if entry.is_dir():
            for r in run_test(entry, bin_path, differ):
                yield r


def main() -> int:
    parser = argparse.ArgumentParser(description="Run black box tests on HVM.")

    parser.add_argument(
        "-b", "--build", help="Compile a new version of HVM.", action="store_true"
    )
    parser.add_argument(
        "-c", "--clean", help="Remove the target directory.", action="store_true"
    )

    args = parser.parse_args()

    if args.clean:
        os.system("cargo clean")

    if args.build:
        os.system("cargo build")

    results = list(run_tests())
    ok = all(map(lambda x: x.ok, results))
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
