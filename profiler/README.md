Profiling
---------

#### 1. Install Dependencies

These are optional dependencies to generate the flowchart output (the flamegraph output doesn't require them):

* [pprof](https://github.com/google/pprof)
* [GraphViz](https://graphviz.org/)

#### 2. Profile

In a shell, run:
```sh
cargo run --profile bench   # To generate the flamegraph and protobuf outputs.
# And optionally:
pprof -svg profile.pb # To generate the flowchart from the protobuf output.
```
