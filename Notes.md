<h1>Notes from Andrej Karpathy's building micrograd video.</h1>
<p>
Date: 13/02/2024
Abhishek Mishra
</p>

- [1. About these Notes](#1-about-these-notes)
- [2. Part 0: micrograd overview](#2-part-0-micrograd-overview)
  - [2.1. Neural Networks](#21-neural-networks)
- [Part 1: Derivative of a simple function with one input](#part-1-derivative-of-a-simple-function-with-one-input)

# 1. About these Notes
Recently I came across Andrej Karpathy's 
["building micrograd" video on youtube][1], after reading a mention of it on 
hackernews perhaps 🤔 (not sure).

I watched the whole video first. Then I was so intrigued I decided to implement
the same engine in a language which is *not python*, so that I can work through
the development of the engine and get it working and also test the results with 
Andrej's version.

In the second pass I watched the video and took notes about what Andrej was
explaining as well as about the python code.

In the third and later passes I slowly implemented the code for each section
and added the code and the results back to this document.

[1]: https://www.youtube.com/watch?v=VMj-3S1tku0

# 2. Part 0: micrograd overview
In this section of tutorial, Andrej provides an overview of micrograd. It is
an autograd engine. It implments backpropagation (reverse mode autodiff) over
a dynamically built DAG.
It is also a small neural networks library with a PyTorch-like API
Micrograd basically allows you to build out mathematical expressions,
and he shows us an example (from the README.md of micrograd).

The library builds an expression and through a forward pass calculates
the value of the expression. It then uses backpropagation to calculate
the gradients of the expression with respect to the input variables.

## 2.1. Neural Networks
Are just mathematical expressions
Take the weights of the neural network and input data as input, and produce
and output.
backpropagation is more general than neural networks, it works with any
mathematical expression.
Finally, micrograd is built using scalars, which is inefficient, but
simplifies the implmentation and allows us to understand the backpropagation 
and the chain rule.
When we want to train a larger network we should be using Tensors.
Andrej's claim is that micrograd is complete. It has only two files engine.py
which knows nothing about neural networks, and nn.py which is a neural
network library built on top of engine.py.
engine.py is literally 100 lines of code in Python. And nn.py is just 60
lines and is a total joke (sic).

* There's a lot to efficiency, but you can get to a working neural network all
  in less than 200 lines of code.

# Part 1: Derivative of a simple function with one input

* Lets get a very good intuitive understanding of what a derivative is.
* Lets define a scalar valued function f(x), and get its value.

```lua
function f(x)
    return (3*(x^2)) - (4*x) + 5
end

f(3.0)
-- 20.0
```

* We can also plot this function over a range of values.

```lua
for x = -5,5,0.25 do
    print(x, f(x))
end

-- values plot given below
```

![plot#0: f(x) over x](plots/plot0.png)