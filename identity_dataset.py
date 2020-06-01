from tensorflow.python.data.ops import dataset_ops
import tensorflow as tf
custom_op = tf.load_op_library("./identity_dataset_op.so")

class IdentityDatasetV2(dataset_ops.UnaryUnchangedStructureDataset):

  def __init__(self, input_dataset):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    variant_tensor = custom_op.identity_dataset(
        self._input_dataset._as_variant_tensor(),
        **self._flat_structure)
    super(IdentityDatasetV2, self).__init__(input_dataset, variant_tensor)

def IdentityDataset(ds):
  return dataset_ops.DatasetV1Adapter(IdentityDatasetV2(ds))
