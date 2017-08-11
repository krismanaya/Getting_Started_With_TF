### Agenda 
"""
Basic operations
Tensor Types
Project speed dating
Placeholders and feeding inputs
Lazy loading

"""


import tensorflow as tf 

### constant 
a = tf.constant(2)
b = tf.constant(3)

### mathematical operation 
x = tf.add(a,b)

### output data 
with tf.Session() as session: 
    data = session.run(x)
    print "addition: 2 + 3 = {}".format(data)


### Visualize is with TensorBoard 

### constant 
a = tf.constant(2, name = "a")
b = tf.constant(3, name = "b")

### mathematical operation 
x = tf.add(a,b, name = "add")

### output data 
with tf.Session() as session: 
    ### add this line to use TensorBoard. 

    writer = tf.summary.FileWriter("./graphs", session.graph)
    data = session.run(x)
    
    print "addition: 2 + 3 = {}".format(data)

### close file "writer" when done using
writer.close()


### Run Visualization

"""
$ python [yourprogram].py
$ tensorboard --logdir="./graphs" -- 6006

--> Then open browser: http://localhost:6006/s

To explicitly name the nodes, 

a = tf.constant(2, name = "a")
b = tf.constant(3, name = "b")

x = tf.add(a, b name = "add")

tf.constant(value, dtype=None, shape=None, name="Const", verify_shape = False)
"""


### More Constants


### these operations add and multiply component wise. 
a = tf.constant([2,2], name = "a")
b = tf.constant([[0,1],[2,3]], name = "b")
x = tf.add(a, b, name = "add")
y = tf.multiply(a, b, name = "multiply")

with tf.Session() as session: 

    x,y = session.run([x,y])
    print "This is addition {}".format(x)
    print "This is Multiplication {}".format(y)


### sepecified value 

### creates a shape of 2 x 2 2d tensor of zeros with floating points
zeros_float = tf.zeros([2,2], dtype=tf.float32, name=None)

with tf.Session() as session: 
    ### run the 2d tensor
    print "creats 2x2 zeros tensor with float type {}".format(session.run(zeros_float))

### creates a tensor of shape and type as the input_tensor but all elem are zeros 
### meaning outputs similar shape of input_tensor as zero wise elements 

input_tensor = [[0,1],[2,3],[4,5]]
zeros_like = tf.zeros_like(input_tensor, dtype = None, name= None, optimize= True)

with tf.Session() as session: 
    print "creates element input wise tensor {}".format(session.run(zeros_like))

### similar argument above but with ones in elements
ones = tf.ones([2,2], dtype = tf.int32, name = "None")
ones_like = tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

with tf.Session() as session: 
    ones, ones_like = session.run([ones, ones_like])
    print "2 x 2 ones tensor {}".format(ones)
    print "ones tensor but input_tensor shape {}".format(ones_like)


### create tensor filled with scalar values 

fill = tf.fill([2,2], 8)  ### ---> [[8,8],[8,8]]

### constants as sequences 

### first things first tensor objects are not iterable

linspace = tf.linspace(1.0, 5.0, 5)
range = tf.range(3, 18, delta = 3, dtype = None, name = "range") ### --> [3,6,9,12,15]
range_2 = tf.range(5)

with tf.Session() as session: 
    linspace, range, range_2 = session.run([linspace, range, range_2])
    print "here is linspace: {}, range: {}, range with only limit: {}".format(linspace, range, range_2)


### Randomly Generated Constants

"""
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
tf.set_random(seed)

"""


