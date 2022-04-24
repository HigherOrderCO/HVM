#!/usr/bin/env python3

import os
from pathlib import Path
import argparse
import subprocess
import platform

from typing import Iterator, Tuple
from dataclasses import dataclass
import difflib
import json

# TODO: run cases in parallel?


@dataclass
class TestResult:
    test_name: str
    case_name: str
    interpreted: bool
    compiled: bool
    ok: bool


is_windows = platform.system() == "Windows"


def compile_test(test_name: str, folder_path: Path, bin_path: Path,
                 code_path: Path) -> Tuple[bool, Path]:
    hvm_comp_cmd = [str(bin_path.absolute()), "compile", str(
        code_path.absolute())]

    # FIXME: Now HVM is not able to run using multiple threads on Windows
    if is_windows:
        hvm_comp_cmd.append("--single-thread")

    p = subprocess.run(
        hvm_comp_cmd, capture_output=True)

    successful_comp = p.returncode == 0

    if successful_comp:
        c_path = folder_path.joinpath(f"{test_name}.c")
        exec_path = folder_path.joinpath(f"{test_name}.out")
        c_comp_cmd = ["clang", str(c_path.absolute()), "-o",
                      str(exec_path.absolute())]

        if not is_windows:
            c_comp_cmd.append("-pthread")

        p = subprocess.run(
            c_comp_cmd, capture_output=True)
        successful_comp = p.returncode == 0
        if not successful_comp:
            print(f"🚨 Clang failed to compile {test_name}")
            print(p.stderr.decode("utf-8").strip())
            return False, None
    else:
        print(f"🚨 HVM failed to compile {test_name}")
        print(p.stderr.decode("utf-8").strip())
        return False, None
    return True, exec_path


def run_test_case(mode: str, case_name: str, bin_path: Path,
                  differ: difflib.Differ, case_args: str,
                  expected_out: str,  code_path: Path = None) -> bool:
    assert mode in (
        "interpreted", "compiled"), 'Mode should be "interpreted" or "compiled"!'

    print(f"Testing case '{case_name}'({mode})... ".ljust(45), end="")

    if mode == "interpreted":
        cmd = [str(bin_path.absolute()), "run", str(
            code_path.absolute()), case_args]
    else:
        cmd = [str(bin_path.absolute()), case_args]

    p = subprocess.run(cmd, capture_output=True)
    test_out = p.stdout.decode("utf-8").strip()

    diff = differ.compare(
        test_out.splitlines(keepends=True),
        expected_out.splitlines(keepends=True),
    )

    diff_lines = [line for line in diff if not line.startswith("  ")]
    if not diff_lines:
        print("✅ PASS")
        return True
    else:
        print("❌ FAILED")
        for line in diff_lines:
            print(line)
        return False


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

    successful_comp, exec_path = compile_test(
        test_name, folder_path, bin_path, code_path)

    specs_path = folder_path.joinpath(f"{test_name}.json")
    assert specs_path.exists(), f"{test_name} test case must have an I/O file!"

    with open(specs_path.absolute(), "r") as jf:
        specs = json.load(jf)
    for case_name in specs:
        spec = specs[case_name]
        case_args = spec["input"]
        expected_out = spec["output"]

        int_suc = run_test_case("interpreted", case_name, bin_path,
                                differ, case_args, expected_out, code_path)
        if successful_comp:
            comp_suc = run_test_case("compiled", case_name, exec_path,
                                     differ, case_args, expected_out)
        else:
            comp_suc = False

        yield TestResult(test_name, case_name, int_suc, comp_suc,
                         int_suc and comp_suc)
    
    if successful_comp:
        os.remove(exec_path)
        os.remove(folder_path.joinpath(f"{test_name}.c"))


def run_tests() -> Iterator[TestResult]:
    differ = difflib.Differ()

    tests_folder = Path(__file__).parent.resolve()

    bin_ext = ".exe" if is_windows else ""
    bin_path = tests_folder.parent.joinpath("target", "debug", f"hvm{bin_ext}")
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
