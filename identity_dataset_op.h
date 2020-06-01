#ifndef TENSORFLOW_CUSTOM_IDENTITY_DATASET_OPS_H_
#define TENSORFLOW_CUSTOM_IDENTITY_DATASET_OPS_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {

REGISTER_OP("IdentityDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

namespace data {

class IdentityDatasetOp : public DatasetOpKernel {
 public:
  explicit IdentityDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
 private:
  class Dataset;
};

}  // namespace data
}  // namespace tensorflow

#endif