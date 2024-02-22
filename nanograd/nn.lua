--- nn.lua: Classes to implement a neural network similar to the micrograd API.
--
-- Date: 22/02/2024
-- Author: Abhishek Mishra

local class = require 'lib/middleclass'
local Value = require 'nanograd/engine'

local nn = {}

local Neuron = class('Neuron')

--- constructor of a Neuron
-- @param nin number of inputs
function Neuron:initialize(nin)
    --- create a random number in the range [-1, 1]
    local function rand_float()
        return (math.random() - 0.5) * 2
    end

    -- create a table of random weights
    self.w = {}
    for _ = 1, nin do
        table.insert(self.w, Value(rand_float()))
    end

    -- create a random bias
    self.b = Value(rand_float())
end

--- forward pass of the Neuron
-- calculate the activation and then apply the activation function
-- which in our case is the tanh function
-- @param x input vector
function Neuron:__call(x)
    local act = self.b
    for i = 1, #self.w do
        act = act + self.w[i] * x[i]
    end
    local out = act:tanh()
    return out
end

local Layer = class('Layer')

--- constructor of a Layer
-- @param nin number of inputs
-- @param nout number of outputs
function Layer:initialize(nin, nout)
    self.neurons = {}
    for _ = 1, nout do
        table.insert(self.neurons, Neuron(nin))
    end
end

--- forward pass of the Layer
-- @param x input vector
function Layer:__call(x)
    local outs = {}
    for _, neuron in ipairs(self.neurons) do
        table.insert(outs, neuron(x))
    end
    return outs
end

nn.Neuron = Neuron
nn.Layer = Layer


-- Tests
-- local n = Neuron(3)
-- local x = { Value(1), Value(2), Value(3) }
-- local y = n(x)
-- print(y)
-- -- Expected output: A Value object with value in the range [-1, 1]

-- local l = Layer(2, 3)
-- local x = { Value(1), Value(2) }
-- local y = l(x)
-- for _, v in ipairs(y) do
--     print(v)
-- end
-- -- Expected output: A table of Value objects with value in the range [-1, 1]

-- export the nn module
return nn
