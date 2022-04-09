#!/usr/bin/env python

import argparse
import os
import subprocess
from pathlib import Path
import difflib
import json


def run_tests():
    differ = difflib.Differ()

    script_path = Path(__file__).parent.resolve()

    hvm_path = script_path.parent.parent.joinpath("target", "debug", "hvm")

    assert hvm_path.is_file(), "HVM must be already compiled"

    for entry in script_path.iterdir():
        if entry.is_dir():
            test_name = entry.name
            test_path = entry.absolute()

            print("TEST:", test_name)

            code_path = test_path.joinpath(f"{test_name}.hvm")
            assert code_path.exists(), f"{test_name} test case must have a hvm source code file!"

            io_path = test_path.joinpath(f"{test_name}.json")
            assert io_path.exists(), f"{test_name} test case must have an I/O file!"

            with open(io_path.absolute(), 'r') as jf:
                test_io = json.load(jf)
            for test in test_io:
                print("TEST CASE:", test, "started")
                cmd = f"{hvm_path} r {code_path.absolute()} {test_io[test]['input']}"

                test_out = "".join(subprocess.getoutput(cmd).split('\n')[3:])
                diff = differ.compare(test_out, test_io[test]["output"])
                diff_lines = [line for line in diff if not line.startswith("  ")]
                if not diff_lines:
                    print("TEST CASE:", test, "pass")
                else:
                    for d in diff_lines:
                        print(d)
                    print("TEST CASE:", test, "failed")


def main():
    parser = argparse.ArgumentParser(description='Run black box tests on HVM.')

    parser.add_argument("-b", "--build", help="Compile a new version of HVM.", action="store_true")
    parser.add_argument("-c", "--clean", help="Remove the target directory.", action="store_true")

    args = parser.parse_args()

    if args.clean:
        os.system("cargo clean")

    if args.build:
        os.system("cargo build")

    run_tests()


if __name__ == "__main__":
    main()
