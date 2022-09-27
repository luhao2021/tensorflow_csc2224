#!/bin/bash -e
pip install pip numpy wheel
pip install keras_preprocessing --no-deps

# todo(chenhao) add to bazel script and change the location in build
mkdir -p /tmp/tf/
clang++ -w -c -emit-llvm -I. -O3 tensorflow/core/platform/cus.cc -o /tmp/tf/cus.bc

# bazel build --config=opt --copt=-g --config=noaws --config=nogcp --config=nohdfs --config=nonccl --verbose_failures //tensorflow/tools/pip_package:build_pip_package
bazel build --config=opt --config=noaws --config=nogcp --config=nohdfs --config=nonccl --verbose_failures //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install -U --no-deps /tmp/tensorflow_pkg/tensorflow-2.4.0-cp38-cp38-linux_x86_64.whl
