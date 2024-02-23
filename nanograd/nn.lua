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

--- get the parameters of the Neuron
function Neuron:parameters()
    local params = {}
    for _, w in ipairs(self.w) do
        table.insert(params, w)
    end
    table.insert(params, self.b)
    return params
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
    if #outs == 1 then
        return outs[1]
    end
    return outs
end

--- get the parameters of the Layer
function Layer:parameters()
    local params = {}
    for _, neuron in ipairs(self.neurons) do
        for _, p in ipairs(neuron:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

local MLP = class('MLP')

--- constructor of a Multi-Layer Perceptron
function MLP:initialize(nin, nouts)
    local sz = table.pack(nin, table.unpack(nouts))
    self.layers = {}
    for i = 1, #nouts do
        table.insert(self.layers, Layer(sz[i], sz[i + 1]))
    end
end

--- forward pass of the MLP
-- @param x input vector
function MLP:__call(x)
    local out = x
    for _, layer in ipairs(self.layers) do
        out = layer(out)
    end
    return out
end

--- get the parameters of the MLP
function MLP:parameters()
    local params = {}
    for _, layer in ipairs(self.layers) do
        for _, p in ipairs(layer:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

nn.Neuron = Neuron
nn.Layer = Layer
nn.MLP = MLP

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

local x = {2, 3, -1}
local mlp = MLP(3, { 4, 4, 1 })
local y = mlp(x)
print(y)
-- Expected output: A table of 1 Value object with value in the range [-1, 1]

-- export the nn module
return nn
