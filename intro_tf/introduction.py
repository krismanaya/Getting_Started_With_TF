"""
Introduction to TensorFlow - CPU vs GPU

In this tutorial we will do simple simple matrix multiplication in 
TensorFlow and compare the speed of the GPU to the CPU, the basis for 
why Deep Learning has become state-of-the art in recent years.

src: https://medium.com/@erikhallstrm/hello-world-tensorflow-649b15aed18c


""" 
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies
import time

### This code example creates pairs 
### of random matrices, clocks the 
### multiplication of them depending on size and device placement.

def get_time(maximum_time): 

    ### get device times (gpu, cpu)
    device_times = {
        "/gpu:0": [], 
        "/cpu:0": []
    }

    matrix_sizes = range(500,50000,50)

    ### loop through can make (n x n) shapes
    for size in matrix_sizes: 
        for device_name in device_times.keys(): 

            ### check the time for each size. 
            print("#### Calculating on the" + device_name + "####")
            shape = (size, size)
            data_type = tf.float16

            ### get context of device. 
            with tf.device(device_name): 
                ### create randomized matrices by dot produciton operation. 
                r1 = tf.random_uniform(shape = shape, minval = 0, maxval = 1, dtype = data_type)
                r2 = tf.random_uniform(shape = shape, minval = 0, maxval = 1, dtype = data_type)

                dot_operation = tf.matmul(r2,r1)

            ### open up tf
            with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as session: 
                start_time = time.time() 
                result = session.run(dot_operation)
                time_taken = time.time() - start_time 
                print(result) 
                device_times[device_name].append(time_taken)

            print(device_times)

            if time_taken > maximum_time: 
                return device_times, matrix_sizes


device_times, matrix_sizes = get_time(1.5)

gpu_times = device_times["/gpu:0"]
cpu_times = device_times["/cpu:0"]

plt.plot(matrix_sizes[:len(gpu_times)], gpu_times, "o-")
plt.plot(matrix_sizes[:len(cpu_times)], cpu_times, "o-")
plt.ylabel("Time")
plt.xlabel("Matrix Size")
plt.show()