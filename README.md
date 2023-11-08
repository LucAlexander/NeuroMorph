# NEUROMORPH
This is the documentation for the library as it exists so far. Further details can be seen in the progress reports detailing the development of this system.

## Model Description Language (MDL)
### Header
The header to any model description is enclosed in `//`. The header takes two arguments: a weight initialization function, and a bias initialization. There should only be one header per model description. Both functions may be parametric and can take up to two whitespace separated float parameters.
```
/xavier,zero/
/he,const_flat 0.1/
/uniform 0 1,const_uneven 0.1 0.5/
```
There are all valid headers

### Layers
Any given layer is comprised of a series of tokens enclosed within parenthesis `()`. At the very minimum, every standard layer needs a name, a width, and an activation function.
```
/xavier,zero/
...
(a, 128, <tanh>)
...
```
Activation and loss functions are enclosed with `<>`, and are often parametric such as with `huber_modified` loss, which takes a float parameter.
```
/xavier,zero/
...
(a, 128, <tanh>)
(output, 4, <relu_parametric, 0.6>, <huber_modified, 0.5>)
```
Every model needs an input node, which consists of only a name and a vector width. Every model also needs an output node, consisting of a name, a vector width, an activation function, and a loss function. These are two unique types of layer node. There must be exactly one of each. The input must be the entry point. 
Layers which are consecutive will be linked togethrt
```
/xavier,zero/
(input, 128)
(a, 144, <tanh>)
(b, 256, <sigmoid>)
(output, 16, <relu_parametric, 0.6>, <huber_modified, 0.5>)
```

### Split
Models can be split into parallel branches. This is done with a special node enclose in brackets `[]`. Divergent nodes are not layers, and cannot have vector widths or functions, instead their only argument other than their name is another model description, not beginning with an input, and not necessarily terminating with an output node. There can be as many branches as desired from any given divergent node. Branches are divided by the pipe `|` operator.
```
/xavier,zero/
(input, 128)
[split, 
    (c, 12, <linear>)(d, 12, <linear>)
|
    (e, 44, <softmax>)
]
(a, 144, <tanh>)
(b, 256, <sigmoid>)
(output, 16, <relu_parametric, 0.6>, <huber_modified, 0.5>)
```

### Merge
Split ends can be converged with a convergence operation back to either the main branch or another parallel branch with another special node denoted by braces `{}`. This node is also not a layer, but it does process data. A convergent node can only take a name, the name of another layer, and require the name of a convergence operation with which to combine the two incoming vectors before being paassed to the next layer or node. The layer chosen to merge with the current branch can be any layer which is not part of the main branch.

These convergences can be forward or backward in time.
```
/xavier,zero/
(input, 128)
{time_travel, other, average}
[split, 
    (c, 12, <linear>)
    {branch_merge, e, multiplicative}
    (d, 12, <linear>)
|
    (e, 44, <softmax>)
]
(a, 144, <tanh>)
(b, 256, <sigmoid>)
[link, [other,]]
{merge, d, additive}
(output, 16, <relu_parametric, 0.6>, <huber_modified, 0.5>)
```

## Install
After downloading the repository, you can run create a virtual environment and use the makefile:
```bash
python3 -m venv env
source env/bin/activate
make clean
make build
```


Or you can run the install commands yourself for a pip candidate:
```bash
python3 -m venv envn
source env/bin/activate
python3 setup.py build
python3 setup.py sdist bdist_wheel
pip install dist/*.whl --force-reinstall
```

Maybe I'll put it on pip in the future.

In any case you can then import the module into any python file:
```python
import neuromorph as nm
```


## Create Models
You can compile any valid MDL string and get the ID of a model in memory. Note that you cannot use this model for anything yet it is only an intermediate representation at this point.
```python
batch_size = 5
learning_rate
input_size = 128
output_size = 4

mdl = f"/xavier,zero/ (input, {input_size})(a, 256, <tanh>)(output, {output_size}, <linear>, <mse>)"
model = nm.compile(mdl, batch_size, learning_rate)

with open("lstm-model", "r") as infile:
    loaded = nm.compile(infile.read(), batch_size, learning_rate)
```

You can set a seed for the random number generator used for the initialization of learnable parameters. Dont worry about this at compile time, weights and biases are determined at build time.
```python
nm.seed(349857)
```

Building is as simple as a function call, this must be done after generating an intermediate representation with `neuromorph.compile`;
```python
nm.build(model)
```

samples = 1000
generate_tensor = (lambda vector_size: [
        [
            [random.random() for i in range(vector_size)]
            for batch in range(batch_size)
        ]
        for k in range(samples)
    ]
)
input_data = generate_tensor(input_size)
expected_data = generate_tensor(output_size)
nm.train(model, input_data, expected_data, 2)
nm.release(model)


## Train
You can then use built models to train on the data of your problem. The train function takes the model ID, an input tensor, a tensor for the expected output values associated with the input tensor values, and a verbosity level. The tensors must be python lists in the shape (sample_count, batch_size, vector_size). The verbosity ranges from 0 (no output) to 2 (extensive output). Anything higher is truncated to 2.
```python
samples = 1000
generate_tensor = (lambda vector_size: [
        [
            [random.random() for i in range(vector_size)]
            for batch in range(batch_size)
        ]
        for k in range(samples)
    ]
)

input_data = generate_tensor(input_size)
expected_data = generate_tensor(output_size)

nm.train(model, input_data, expected_data, 2)
```


## Cleanup
It is a good idea to release the heap memory associated with the model IDs you have compiled or built during the lifespan of your program.
```python
nm.release(model)
nm.release(loaded)
```

## Example Models
**small-model**
```
/normal 0 0.1,zero/
(input, 4)
{gate, recur, additive}
(a, 4, <relu, 5.9>)
(b, 4, <swish, 5.9>)
[link,[recur,]]
(output, 4, <sigmoid>, <mse, 4>)
```

**big-model**
```
/xavier,const_uneven 0.1 0.3/


(input, 4)
(a, 4, <sigmoid>)
{c2, g, additive}
(b, 4, <relu>)
[b1, 
	(d, 4, <softmax>)
	{standby, stalerecur, additive}
	(e, 4, <softmax>)
|
	(f, 4, <relu>)
	[stale,[stalerecur,]]
	(g, 4, <relu>)
]
(c, 4, <sigmoid>)
{c1, e, additive}
(output, 4, <sigmoid>, <huber_modified, 2.4>)
```

**gated-model**
```
/uniform 0 0.5,const_uneven 0.1 0.2/


(input, 4)
{gate, doubleforget, additive}
(a, 4, <relu>)
(b, 4, <swish, 5.9>)
[link,(forget, 4, <tanh>)(doubleforget, 4, <tanh>)]
(output, 4, <sigmoid>, <mse>)
```

**lstm-model**
```
/xavier,const_flat 0.5/
(input, 4)
{a, lastrecur, additive}
[b,
	(include, 4, <sigmoid>)
|
	(process, 4, <tanh>)
	{pi, include, multiplicative}
|
	(add, 4, <sigmoid>)
]
(forget, 4, <sigmoid>)
{state0, prevstaterecur, multiplicative}
{state1, pi, additive}
[prevstate,[prevstaterecur,]]
(statep, 4, <tanh>)
{lastc, add, multiplicative}
[last,[lastrecur,]]
(output, 4, <tanh>, <mse>)
```

# TODO
* Serialization
* Using trained models
* refactors and optimizations
* continue SIMD for different architectures / versions
