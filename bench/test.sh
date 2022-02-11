#!/bin/bash

mkdir "~test"

failed=

for dir in *; do

  p="${dir}/test.txt"
  if [ -d "${dir}" ] && [ -f "${p}" ]; then
    echo "Testing '${dir}'"

    bin_path="~test/${dir}"

    hvm compile "${dir}/main.hvm"
    clang -O2 "./${dir}/main.c" -o "${bin_path}" -lpthread

    while read -r line; do
      IFS=' ' read -ra arr <<< "$line"
      output="${arr[0]}"
      input=("${arr[@]:1}")
      printf "testing input: %s ... " "${input[*]}"
      # printf "(expected output: %s) " "${output}" # DEBUG

      result="$("${bin_path}" "${input[@]}" 2>/dev/null)"
      # printf "<<%s>> " "$result" # DEBUG

      if [ "$result" == "$output" ]; then
        echo "OK"
      else
        failed=1
        echo "XXX FAILED XXX"
      fi

    done < "${p}"
  fi

done

if [ -n "${failed}" ]; then
  exit 1
fi
