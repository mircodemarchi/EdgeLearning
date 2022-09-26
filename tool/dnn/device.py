import tensorflow as tf
devices = tf.config.list_physical_devices()
print(devices)

# #Use only CPU
# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

with tf.device("/cpu:0"):
    devices = tf.config.list_physical_devices()
    print(devices)
    tf.debugging.set_log_device_placement(True)
    a=tf.random.normal([100,100])
    b=tf.random.normal([100,100])
    print(a[0, :])
    print(b[:, 0])
    c = a*b
    print(c[0, 0])