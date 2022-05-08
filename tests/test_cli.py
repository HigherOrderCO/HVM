#!/usr/bin/env python3

# TODO:
# - single thread mode flag ?
# - multiple compilers
#   - gcc
#   - tcc
# - pthreads on Windows

from pathlib import Path
import argparse
import subprocess
import platform
from typing import Any, Iterator, List, Literal, Optional, Union
from dataclasses import dataclass
from difflib import Differ
import json

is_windows = platform.system() == "Windows"

C_COMPILER = "clang"


def c_compiler_cmd(in_path: str, out_path: str) -> List[str]:
    cmd = [C_COMPILER]
    if not is_windows:
        cmd.append("-lpthread")
    cmd += [in_path, "-o", out_path]
    return cmd


@dataclass
class Compiled:
    program_path: Path


@dataclass
class Interpreted:
    hvm_cmd: str
    program_path: Path


TestMode = Union[Compiled, Interpreted]
TestModeStr = Union[Literal["compiled"], Literal["interpreted"]]


@dataclass
class TestResult:
    mode_str: TestModeStr
    test_name: str
    case_name: str
    ok: bool


def get_mode_str(mode: TestMode) -> TestModeStr:
    match mode:
        case Compiled(_):
            return "compiled"
        case Interpreted(_, _):
            return "interpreted"


def resolve_path(path: Path) -> str:
    return str(path.absolute())


def assert_file(path: Path, err: str):
    assert path.is_file(), err


def main() -> int:
    parser = argparse.ArgumentParser(description="Run black box tests on HVM.")

    parser.add_argument(
        "--hvm-cmd", type=str, default="hvm"
    )
    parser.add_argument(
        "--run-mode", choices=["compiled", "interpreted"], action="append"
    )

    args = parser.parse_args()

    modes = args.run_mode or ["interpreted"]
    hvm_cmd: str = args.hvm_cmd

    exit_code = 0

    for mode in modes:
        assert mode in ("compiled", "interpreted")
        results = list(run_tests(mode, hvm_cmd))
        ok = all(map(lambda x: x.ok, results))
        if not ok:
            exit_code = 1

    return exit_code


def run_tests(mode_str: TestModeStr, hvm_cmd: str) -> Iterator[TestResult]:
    differ = Differ()

    base_test_folder = Path(__file__).parent.resolve()


    test_folders = list(filter(lambda x: x.is_dir(), base_test_folder.iterdir()))
    for entry in test_folders:
        yield from run_test(differ, mode_str, hvm_cmd, entry)


def run_test(
    differ: Differ, mode_txt: TestModeStr, hvm_cmd: str, folder_path: Path
) -> Iterator[TestResult]:
    test_name = folder_path.name
    folder_path = folder_path.absolute()

    print()
    print(f"Testing: {test_name}")

    code_path = folder_path.joinpath(f"{test_name}.hvm")
    assert_file(
        code_path, f"'{test_name}' case must have an HVM file at '{code_path}'!"
    )

    spec_path = folder_path.joinpath(f"{test_name}.json")
    assert_file(
        spec_path, f"'{test_name}' case must have a spec file at '{spec_path}'!"
    )

    with open(spec_path.absolute(), "r") as jf:
        specs = json.load(jf)

    match mode_txt:
        case "interpreted":
            mode = Interpreted(hvm_cmd, code_path)
            yield from run_cases(differ, mode, test_name, specs)
        case "compiled":
            exec_path = compile_test(test_name, folder_path, hvm_cmd, code_path)
            if exec_path is None:
                yield TestResult(mode_txt, test_name, "*", False)
            else:
                mode = Compiled(exec_path)
                yield from run_cases(differ, mode, test_name, specs)

                exec_path.unlink(missing_ok=True)
                folder_path.joinpath(f"{test_name}.c").unlink(missing_ok=True)


def run_cases(differ: Differ, mode: TestMode, test_name: str, specs: Any):
    for case_name, spec in specs.items():
        case_args = spec["input"]
        expected_out = spec["output"]

        success = run_test_case(
            differ,
            mode,
            case_name,
            case_args,
            expected_out,
        )
        yield TestResult(get_mode_str(mode), test_name, case_name, success)


def run_test_case(
    differ: Differ,
    mode: TestMode,
    case_name: str,
    case_args: str,  # TODO: refactor to list
    expected_out: str,
) -> bool:
    mode_txt = get_mode_str(mode)
    print(f"Case '{case_name}' ({mode_txt})... ".ljust(45), end="")

    match mode:
        case Interpreted(hvm_cmd, program_path):
            code_path_abs = resolve_path(program_path)
            cmd = [hvm_cmd, "run", code_path_abs, case_args]
        case Compiled(program_path):
            program_path_abs = resolve_path(program_path)
            cmd = [program_path_abs, case_args]

    p = subprocess.run(cmd, capture_output=True)

    if p.returncode != 0:
        print("❌ FAILED")
        # print(p.stdout.decode('utf-8'))
        print(p.stderr.decode('utf-8'))
        return False

    test_out = p.stdout.decode("utf-8").strip()

    diff = differ.compare(
        expected_out.splitlines(keepends=True),
        test_out.splitlines(keepends=True),
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


def compile_test(
    test_name: str, folder_path: Path, hvm_cmd: str, code_path: Path
) -> Optional[Path]:
    hvm_comp_cmd = [hvm_cmd, "compile", str(code_path.absolute())]

    p = subprocess.run(hvm_comp_cmd, capture_output=True)

    successful_comp = p.returncode == 0

    if not successful_comp:
        print(f"🚨 HVM failed to compile ({test_name})")
        print(p.stderr.decode("utf-8").strip())
        return None
    else:
        c_path = folder_path.joinpath(f"{test_name}.c")
        bin_path = folder_path.joinpath(f"{test_name}.out")
        c_comp_cmd = c_compiler_cmd(resolve_path(c_path), resolve_path(bin_path))

        p = subprocess.run(c_comp_cmd, capture_output=True)
        successful_comp = p.returncode == 0
        if not successful_comp:
            print(f"🚨 Failed to compile C code ({test_name})")
            print(p.stderr.decode("utf-8").strip())
            return None

        return bin_path


if __name__ == "__main__":
    exit(main())
