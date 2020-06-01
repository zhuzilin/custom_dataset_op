# custom dataset op
This repo is an example on how to create custom dataset op in tensorflow, so that an extention on `tf.data` may not need to recompile the whole codebase. The code is tested for v1.15.0.

To make it as simple as possible, we created an `IdentityDatasetOp`, which behaves as an `IdentityOp` but is implemented with the dataset interface tensorflow requires.

## compilation

To run the op, first compile it as the official doc instructed:

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared identity_dataset_op.cc -o identity_dataset_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

There may be some problem with this command, and in that case you can simple append the output of `tf.sysconfig.get_compile_flags()` and `tf.sysconfig.get_link_flags()` at the end. In my case, it is:

```bash
g++ -std=c++11 -shared identity_dataset_op.cc -o identity_dataset_op.so -fPIC -I/usr/local/lib/python3.7/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/usr/local/lib/python3.7/site-packages/tensorflow_core -ltensorflow_framework.1
```

## test

After you have successfully compiled the `.so`, you could run `test_graph.py` and `test_eager.py` to test it. Right now, there is a refcount error with regard to eager execution and I've added an issue to the community ([here](https://github.com/tensorflow/tensorflow/issues/39986))

