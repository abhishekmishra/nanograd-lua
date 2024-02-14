<h1>Notes from Andrej Karpathy's building micrograd video.</h1>
<p>
Date: 13/02/2024
Abhishek Mishra
</p>

<hr/>

**Table of Contents**

- [1. About these Notes](#1-about-these-notes)
- [2. Part 0: micrograd overview](#2-part-0-micrograd-overview)
  - [2.1. Neural Networks](#21-neural-networks)
- [3. Part 1: Derivative of a simple function with one input](#3-part-1-derivative-of-a-simple-function-with-one-input)
  - [3.1. Derivative of the Expression](#31-derivative-of-the-expression)
  - [3.2. Derivative at Another Point (x = -3)](#32-derivative-at-another-point-x---3)
  - [3.3. Derivative goes to 0](#33-derivative-goes-to-0)
- [4. Part 2: A More Complex Case](#4-part-2-a-more-complex-case)
- [5. Expressions for Neural Networks](#5-expressions-for-neural-networks)
  - [5.1. Core Value Object](#51-core-value-object)
  - [5.2. Addition of Value Objects](#52-addition-of-value-objects)
  - [5.3. Multiplication of Value Objects](#53-multiplication-of-value-objects)
  - [5.4. Children of Value Objects](#54-children-of-value-objects)
- [6. References](#6-references)

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

# 3. Part 1: Derivative of a simple function with one input

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

## 3.1. Derivative of the Expression
* Now we will think about the derivative of the expression.
* See the [Differentiation rules][2]
* In neural networks no one actually writes an expression and derives it.
* We are not going for the symbolic approach.
* We will try and understand what the derivative is measuring and what it is
  telling us about the function.
* We look at the definition of the derivative in terms of Limit from the wiki
  page of derivative.
TODO: update the definition of derivative here.
* Basically how does the function respond to an infinitesimal change in the
  input variable. What is the slope of the function at the point.

```lua
-- if we use too small h, we will eventuall get an incorrect value because
-- we are using floating point arithmetic.
h = 0.00001
x = 3.0

f(x+h)
-- 20.014003

(f(x+h) - f(x))/h
-- 14.003000000002
```

* From the above we can conclude that at x=3 the slope of f(x) is 14.
* We can also calculate using the derivative of f(x).
* f'(x) or df(x)/dx = 6*x - 4
* Therefore f'(x) at x = 3 is 14.

## 3.2. Derivative at Another Point (x = -3)

* Let's calculate slope at another point, say x = -3
* Even looking at the plot we can see that the slope of the function at x = -3
  is negative. Therefore the sign of the slope will be 'minus'.
* Slope or f'(-3) is -22.

```lua
x = -3

(f(x+h) - f(x))/h
-- -21.999970000053
```

## 3.3. Derivative goes to 0

* At x=2/3, the function's slope is 0.
* So the function will not respond to a nudge at this point.

```lua
x = 2/3

(f(x+h) - f(x))/h
-- 3.0000002482211e-05
```

# 4. Part 2: A More Complex Case

* Let's take a function with more than one inputs.
* We consider a function with three scalar inputs - a, b, c with a single output
  d.

```lua
a = 2.0
b = -3.0
c = 10.0

d = a*b + c

d
-- 4.0
```

* Now we would like to get the derivative of d w.r.t a, b and c.
* We would like to get the intuition of what this will look like.
* Lets start with derivative with respect to a. This means we will change a by
  a small amount and calculate d. And then we will calculate slope at the point.
* The value of d reduces by a small amount when we increase h by a small amount,
  as a is multiplied by b in the expression, and b is negative. Thus increase
  in a decreases the value of d.
* This gives us an intuition about the slope of d with respect to a.
* Note that using rules of differentiation also we will get the same answer
  as the calculation below.
* d(d)/da = b; therefore slope is b = -3.0

```lua
h = 0.00001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c

a = a + h

d2 = a*b + c

print('d1 = ' .. d1)
-- d1 = 4.0
print('d2 = ' .. d2)
-- d2 = 3.99997
print('slope = ' .. (d2 - d1)/h)
-- slope = -3.0000000000641
```

* Now lets consider the derivative of d w.r.t. b.
* Again from the rules of differentiation d(d)/db = a.
* Therefore we should expect the answer 2.0.

```lua
h = 0.00001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c

b = b + h

d2 = a*b + c

print('d1 = ' .. d1)
-- d1 = 4.0
print('d2 = ' .. d2)
-- d2 = 4.00002
print('slope = ' .. (d2 - d1)/h)
-- slope = 2.0000000000131
```

* Finally lets consider the derivative of d w.r.t. c.
* From the rules of differentiation d(d)/dc = 1.
* With changes in c, d changes by the exact same amount.

```lua
h = 0.00001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c

c = c + h

d2 = a*b + c

print('d1 = ' .. d1)
-- d1 = 4.0
print('d2 = ' .. d2)
-- d2 = 4.00001
print('slope = ' .. (d2 - d1)/h)
-- slope = 0.99999999996214
```

* We have some intuitions about how expressions and their derivatives will work.
* Lets move to neural networks which will have massive expressions.

# 5. Expressions for Neural Networks

As mentioned neural networks will have massive expressions. So we need some
datastructure to maintain the massive expressions. And so we will build out the
`Value` object which was shown in the beginning of the video, from the README
of the micrograd project.

## 5.1. Core Value Object

* Lets start with the skeleton of a very simple value object.
* *Lua Note:* Lua is object-oriented but does not have classes. To keep the
  structure of the code similar to the one in the video,
  we will write the classes using the excellent [*middleclass*][3] library.
    - The code will be slightly more verbose than python.
* Here we create a simple value class, then create an instance `a`, and
  finally print it out.
* *Lua Note:* To make sure the code can be run in an interpreter, all Lua
  variables are being created in global scope. Usually we would write the code
  in files, and make sure that the variables are marked `local`.

```lua
class = require 'middleclass'

-- Declare the class Value
Value = class('Value') -- 'Value' is the class' name

-- constructor
function Value:initialize(data)
  self.data = data
end

-- tostring
function Value:__tostring()
  return 'Value(data = ' .. self.data .. ')'
end

a = Value(2.0)

a
-- Value(data = 2.0)
```

## 5.2. Addition of Value Objects

* Now, we would like to create mutliple values and also be able to do things
like `a + b` where `a` and `b` are values.
* We're going to use the metamethod `__add` in Lua to allow us to define 
  addtion for Value objects.
* The addition inside `Value:__add` is a simple floating point addition of the
  data of two Value objects.

```lua
Value = class('Value')

function Value:initialize(data)
  self.data = data
end

function Value:__tostring()
  return 'Value(data = ' .. self.data .. ')'
end

-- add this Value object with another
-- using metamethod _add
function Value:__add(other)
  return Value(self.data + other.data)
end

a = Value(2.0)
b = Value(-3.0)

-- this line will invoke the metamethod Value:__add
a + b
-- Value(data = -1.0)

```

## 5.3. Multiplication of Value Objects

* Multiplication of Value objects is fairly simple and uses the `__mul`
  metamethod.
* This will now help us write expressions like `a * b` and `a * b + c`.

```lua
-- Class definition same as in the previous snippet.
-- multiply this Value object with another
-- using metamethod _mul
function Value:__mul(other)
  return Value(self.data * other.data)
end

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

a * b
-- Value(data = -6.0)

-- the next line is equivalent to
-- (a.__mul(b)).__add(c)
d = a * b + c

d
--Value(data = 4.0)
```

## 5.4. Children of Value Objects

* What we're missing is the connective tissue of the expression.
* We want to keep these expression graphs, so we need to keep pointers about
  what values produce what other values.
* So we're going to introduce a new variable called `_children` which will be
  by default an empty tuple.
* *Lua Note:* Lua does not have tuples. In fact it has only one in-built
  compound datatype **tables**. So we're going to use a table to store
  `_children`.
* Internally the `children` are stored as **set** for efficiency.
* *Lua Note:* Lua does not have sets either. However sets can be eumulated in
  Lua using tables by keeping the elements as *indices* of a table. See 
  [11.5 – Sets and Bags (Programming in Lua)][4] for details of this approach.

```lua
class = require 'middleclass'
Set = require 'set'

function merge_tables(first, second)
  for k,v in pairs(second_table) do first_table[k] = v end
end

Value = class('Value')

function Value:initialize(data, _children)
  self.data = data
  if _children == nil then
    self._prev = Set.empty()
  else
    self._prev = Set(_children)
  end
end

function Value:__tostring()
  return 'Value(data = ' .. self.data .. ')'
end

function Value:__add(other)
  return Value(self.data + other.data, {self, other})
end

function Value:__mul(other)
  return Value(self.data * other.data, {self, other})
end

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c

d._prev
-- {Value(data = -6.0), Value(data = 10.0)}

```

# 6. References

[1]: https://www.youtube.com/watch?v=VMj-3S1tku0
[2]: https://en.wikipedia.org/wiki/Differentiation_rules
[3]: https://github.com/kikito/middleclass
[4]: https://www.lua.org/pil/11.5.html