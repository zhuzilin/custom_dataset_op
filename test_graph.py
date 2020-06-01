import tensorflow as tf
from identity_dataset import IdentityDataset

if __name__ == "__main__": 
  ds = tf.data.Dataset.from_tensor_slices([1, 2, 3]) 
  ds = IdentityDataset(ds)
  i = ds.make_initializable_iterator() 
 
  sess = tf.Session() 
 
  sess.run(i.initializer) 
 
  for _ in range(3): 
    print(sess.run(i.get_next()))
