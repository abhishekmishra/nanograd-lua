--- nn.lua: Classes to implement a neural network similar to the micrograd API.
--
-- Date: 22/02/2024
-- Author: Abhishek Mishra

local class = require 'lib/middleclass'
local Value = require 'nanograd/engine'

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

-- Tests
-- local n = Neuron(3)
-- local x = { Value(1), Value(2), Value(3) }
-- local y = n(x)
-- print(y)
-- -- Expected output: A Value object with value in the range [-1, 1]