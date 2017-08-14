
### What is TensforFlow? 
"""

* Open Source Software Library for numerical computation using data 
  flow graphs 

* Originally devloped by Google Brain Team to conduct ML and DNN research 

* General enough to be applicable in a wide variety of other domains as well. 

Definition Tensor: 

An n-dimensional array 

0d tensor: scalar 
1d tensor: vector 
2d tensor: Matrix 
.
.
.
. 
nd tensor: n-dimensional tensor 
"""

### Let's get started! 

import tensorflow as tf 

### automatic naming (x,y) instead of x = a, y = b. 

"""
Nodes: Operators, Variables and constants
Edges: Tensors
Read: Read graph source to sink 
DataFlow --> TensorFlow (Tensors are data)

       8
       | (tf.Session())
       |
   (a = sink)
     /   \
   (3)   (5)
"""

a = tf.add(3,5)

### read in data: 
### A Sessions object encapsulates the 
### Environment in which Operation objects 
### are executed, and Tensor objects are evaluated. 

with tf.Session() as session: 

    ### this runs in the node a
    data = session.run(a)
    print "3 + 5 = {}".format(data)

### More Graphs
"""
            (POW = Sink)
           /     \
          /       \ 
(x)      / "scalar"\ 
   \    /           \
    (MUL= Sub_Sink)  \
   /                  \
(y)             (x)--(Add = Sub_Sink) 
                       /
                    (y) 
"""

x = 2 
y = 3 
ss_1 = tf.add(x,y)
ss_2 = tf.multiply(x,y)
sink = tf.pow(ss_2, ss_1)
with tf.Session() as session: 

    data = session.run(sink)
    print("6^5 = {}".format(data))

### Subgraphs
"""
(x) --->(useless)
       /
      / (POW)
     /   /\
(x)  \  /  \         
   \  |/    \
  (ADD)      \
   /        (MUL) 
(y)          /\ 
          (x)  (y)

Becase we only want the value of pow and pow doesnt 
depend on useless --> we can save the computation 
for other needs.
"""
x = 2 
y = 3 

add = tf.add(x,y)
mul = tf.multiply(x,y)
useless = tf.multiply(x, add)
power = tf.pow(add, mul)

with tf.Session() as session: 
    ### tf.Session.run(fetches, feed_dict = None, 
    ###                option = None, run_metadata = None)

    data, not_useless = session.run([power, useless])
    print("Now using {} and power add^mul = {}".format(not_useless,data))

### Distributed Computation
"""
Possible to break graphs into seneral chunks 
and run them parallelly 
across multiple CPUs, GPUs, or devices. 
"""

### Creates a graph. 
with tf.device('/cpu:0'): 
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a', shape = [1,6])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b', shape = [6,1])
    c = tf.matmul(a,b)

### Creates a session log_device_placement set to True. 
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session: 
    data = session.run(c)
    print("Session with log_device_placement {}").format(data)



### Make a graph 

g = tf.Graph() 

### add operations set as default

with g.as_default(): 

    x = tf.add(3,5)

sess = tf.Session(graph = g)

sess.close()

### handles default graph 

g = tf.get_default_graph() 

### No more then one graph

"""
g1 = tf.get_default_graph()

g2 = tf.Graph() 

add ops to the default graph 

with g1.as_default(): 

    a = tf.Constant(3)


add ops to the user created graph 

with g2.as_default(): 

    b = tf.Constant(5)
"""

### Why Graphs? 

"""

1. Save computation (only run subgraphs that lead
   to the values you want to fetch)

2. Break computation into small, differential pieces 
   to facilitates auto-differentiation 

3. Facilitate distributed computation, spread the 
   work across multiple CPUs, GPUs, or devices 

4. Many common mahine models are commonly taught 
   and visualized as directed dgraphs already. 

"""
