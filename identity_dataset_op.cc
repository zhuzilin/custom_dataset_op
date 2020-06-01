#include "identity_dataset_op.h"

namespace tensorflow {

namespace data {

class IdentityDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)),
        input_(input) {
    input_->Ref();
  }

  ~Dataset() override {
    input_->Unref();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, "IdentityDatasetIterator"});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return "IdentityDataset";
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return ::tensorflow::errors::Unimplemented(
          "AsGraphDefInternal for IdentityDataset is not implemented yet");
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}
    
    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      return ::tensorflow::errors::Unimplemented(
          "SaveInterval for IdentityDataset is not implemented yet");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return ::tensorflow::errors::Unimplemented(
          "RestoreInternal for IdentityDataset is not implemented yet");
    }
   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
  };  // class Iterator

  const DatasetBase* const input_;
};  // class Dataset

void IdentityDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  *output = new Dataset(ctx, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("IdentityDataset").Device(DEVICE_CPU),
                        IdentityDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
