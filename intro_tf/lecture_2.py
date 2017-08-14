### Agenda 
"""
Basic operations
Tensor Types
Project speed dating
Placeholders and feeding inputs
Lazy loading

"""


import tensorflow as tf 
import numpy as np

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
# with tf.Session() as session: 
#     ### add this line to use TensorBoard. 

#     writer = tf.summary.FileWriter("./graphs", session.graph)
#     data = session.run(x)
    
#     print "addition: 2 + 3 = {}".format(data)

# ### close file "writer" when done using
# writer.close()


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

### Tensor operations 
a = tf.constant([3,6])
b = tf.constant([2,2])

tf.add(a,b) ### --> [(3+2) (6+2)]
tf.add_n([a, b, b]) ### ---> a + b + b  [(3+2+2) (6+2+2)]
tf.multiply(a,b) ### --> [(3*2) (6*2)] because multiplication is element wise 
# tf.matmul(a,b) ### --> ValueError 
tf.matmul(tf.reshape(a, [1,2]), tf.reshape(b,[2,1])) ### --> [[18]] vector dot-product
tf.div(a,b) ### --> [(3 / 2 == 1) (6 / 2) == 3]
tf.mod(a,b) ### --> [1 0]



### Data Types 
"""
0-d tensor, or "scalar"
t_0 = 19
tf.zeros_like(t_0) # ==> 0
tf.ones_like(t_0) # ==> 1

1-d tensor, or "vector"
t_1 = ['apple', 'peach', 'banana']
tf.zeros_like(t_1) # ==> ['' '' '']
tf.ones_like(t_1) # ==> TypeError: Expected string, got 1 of type 'int' instead.

2x2 tensor, or "matrix"
t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]
tf.zeros_like(t_2) # ==> 2x2 tensor, all elements are False
tf.ones_like(t_2) # ==> 2x2 tensor, all elements are True

"""

### 1-d tensor, or "vector"
t_1 = ['apple', 'peach', 'banana']
zeros_like = tf.zeros_like(t_1) # ==> empty strings "" ??? 
# ones_like = tf.ones_like(t_1) # ==> Type error value

with tf.Session() as session: 
    zeros_like = session.run(zeros_like)
    print "zeros_like string {}".format(zeros_like)
    # print "ones_like string {}".format(ones_like)



### Variables

### create a varible with a scalar value 
a = tf.Variable(2, name = "Scalar")

### create a variable with a vec value 
b = tf.Variable([2,3], name = "vector")


### create variable as a 2 x 2 matrix 
c = tf.Variable([[0,1],[2,3]], name="matrix")

### create variable W as 784 x 10 tensor, filled with zeros 
W = tf.Variable(tf.zeros([784,10]))


"""
tf.Variable holds several ops: 

x = tf.Variable(...)
x.initializer ## init op 
x.value() ## read op 
x.assign(...) ## write op
x.assign_add(...) ## and more

"""

##### You Have to initialize your variables ##### 

"""
Initialize all variables at once: 
init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)


Initialize only a subset of variables:
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
     sess.run(init_ab)


Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
  sess.run(W.initializer)
"""

### The easier way to initializing all variables at once: 

init = tf.global_variables_initializer()
with tf.Session() as session: 
    init = session.run(init)
    ### Do some work in the model. 
    # ...
    # ... 
    # ...
    print "a, b, c , w are inside the session"


### W is a random 700 x 100 variable object 
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as session: 
    session.run(W.initializer)
    session.run(assign_op)
    print W ### prints object
    print W.eval() #### prints Tensor 
    print W.eval()


### create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var")

### assign a * 2 to a and call that op a_time_two 
my_var_times_two = my_var.assign(2*my_var)

with tf.Session() as session: 
    session.run(my_var.initializer)
    """
    the assign key when running will 
    assign 2*my_var to a every time 
    my_var_times_two is fetched. 
    """
    session.run(my_var_times_two) # 4
    session.run(my_var_times_two) # 8
    session.run(my_var_times_two) # 16
    print my_var.eval()



### assign_add(0 and assign_sub() 
my_var = tf.Variable(10, name = "Sacalar")

with tf.Session() as session: 
    session.run(my_var.initializer)

    # increment by 10 
    my_var.assign_add(10) # >> 20

    # decrement by 2 
    my_var.assign_sub(2) # >> 18

"""
Each session matains its own copy 
of Variables EX: 

W = tf.Variable(10)

sess1 = tf.Session() 
sess2 = tf.Session() 
sess1.run(W.initializer)
sess2.run(W.initializer)

print sess1.run(W.assign_add(10)) >> 20
print sess2.run(W.assign_sub(2)) >> 8


This can go on forever 
sess1.close() 
sess2.close() 
"""

### Use a variable to initialize another variable 
### Want to declare U = 2 *W 

### W is a random 700 x 10 tensor 
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(2 * W.initialized_value()) ### Ensure that W is initialized before 


### Session Vs Interactive Session

session = tf.InteractiveSession() 
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b 
### We can just use "c.eval()" without specifying the context "sess"
print("this is c: {}".format(c.eval()))

session.close()


### Control Dependencies 

"""
tf.Graph.control_dependencies(control_inputs)
# defines which ops should be run first
# your graph g have 5 ops: a, b, c, d, e

with g.control_dependencies([a, b, c]):
# 'd' and 'e' will only run after 'a', 'b', and 'c' have executed. 

d = ...
e= ...

 """

### A quick reminder 
"""
A TF program often has 2 phases: 

1. Assemble a graph 
2. Use a session to execute operations in the graph 

--> Can assemble the graph first without knowing the values needed for computation 

Analogy: 
Can Defind the function f(x,y) = x*2 + y without knowing value of x or y. 
x, y are placeholders for the actual values. 

We, or our clients, can later supply their own data when tehy need to 
execute the computation. 

"""

### Placeholders tf.placeholder(dtype, shape=None, name=None)

"""
(const)--->(add)
             /
            /
(placholder)
"""

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements 
b = tf.constant([5,5,5], tf.float32)

# use the place holder as you would a constant or a variabel 
c = a + b # short for tf.add(a,b)

with tf.Session() as session: 
    # feed [1,2,3] to placeholder a via the dict {a: [1,2,3]}
    # fetch value of c 
    data = session.run(c ,{a: [1,2,3]}) # the tensor a is a key, not the string "a"
    print "here is the data feed: {}".format(data)


### What if we want to feed multiple data points in? 
"""
with tf.Session() as session:
    for a_value in list_of_values_for_a: 
        print session.run(c, {a: a_value})
"""

list_of_values_for_a = [[1,2,3],[4,5,6],[7,8,9]]
with tf.Session() as session: 
    for a_value in list_of_values_for_a: 
        print(session.run(c, {a: a_value}))


### You can feed_dict any feedable tensor. 
### placeholder is jsut a way to indicate that 
### something must be fed. 

### tf.Graph.is_feedable(tensor) True iff tensor is feedable. 


### feeding values to TF operations 

# create operations, tensors, etc (using the default graph)
a = tf.add(2,5)
b = tf.multiply(a, 3)

with tf.Session() as session: 
    # define a dictionary that says to replace the value of 'a' with 15
    replace_dict = {a: 15}

    # run the session, passsing in "replace_dict" as the value to "feed_dict"
    print (session.run(b, feed_dict = replace_dict))



### Lazy Loading - defer creating / initializing an object until needed

### Example 


x = tf.Variable(10, name = 'x')
y = tf.Variable(20, name = 'y')
z = tf.add(x,y) # you create teh node for add node before executing graph 

with tf.Session() as session: 
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/12', session.graph)
    for _ in np.arange(10): 
        session.run(z) ### lazy would be adding tf.add(x,y) instead of calling variable
    writer.close() 

"""
tf.get_default_graph().as_graph_def() 

Normal loading: 

node {
  name: "Add"
  op: "Add"
  input: "x/read"
  input: "y/read"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
} }


Lazy Loading: 

node {
  name: "Add"
  op: "Add"
  ...
  }
... node {
  name: "Add_9"
  op: "Add"
  ...
}
"""




