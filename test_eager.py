import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from identity_dataset import IdentityDatasetV2

if __name__ == "__main__":
  tf.enable_eager_execution()
  ds = tf.data.Dataset.from_tensor_slices([1, 2, 3]) 
  ds = IdentityDatasetV2(ds)
 
  for x in ds: 
    print(x)