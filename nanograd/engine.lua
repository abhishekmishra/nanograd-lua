--- engine.lua: Value class for the nanograd library, based on micrograd.
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local class = require 'lib/middleclass'
local Set = require 'util/set'

--- Declare the class Value
Value = class('Value')

--- static incrementing identifier
Value.static._next_id = 0

--- static method to get the next identifier
function Value.static.next_id()
    local next = Value.static._next_id
    Value.static._next_id = Value.static._next_id + 1
    return next
end

--- constructor
function Value:initialize(data, _children, _op, label)
    self.data = data
    self.grad = 0
    self._op = _op or ''
    self.label = label or ''
    self._backward = function() end
    self.id = Value.next_id()
    if _children == nil then
        self._prev = Set.empty()
    else
        self._prev = Set(_children)
    end
end

--- string representation of the Value object
function Value:__tostring()
    return 'Value(data = ' .. self.data .. ')'
end

--- add this Value object with another
-- using metamethod _add
function Value:__add(other)
    local this = self
    if type(other) == 'number' then
        other = Value(other)
    end
    if type(self) == 'number' then
        this = Value(self)
    end

    local out = Value(this.data + other.data, { this, other }, '+')
    local _backward = function()
        this.grad = this.grad + (1 * out.grad)
        other.grad = other.grad + (1 * out.grad)
    end
    out._backward = _backward
    return out
end

--- multiply this Value object with another
-- using metamethod _mul
function Value:__mul(other)
    local this = self
    if type(other) == 'number' then
        other = Value(other)
    end
    if type(self) == 'number' then
        this = Value(self)
    end

    local out = Value(this.data * other.data, { this, other }, '*')
    local _backward = function()
        this.grad = this.grad + (other.data * out.grad)
        other.grad = other.grad + (this.data * out.grad)
    end
    out._backward = _backward
    return out
end

function Value:exp()
    local x = self.data
    local out = Value(math.exp(x), { self }, 'exp')
    local _backward = function()
        -- because the derivative of exp(x) is exp(x)
        -- and out.data = exp(x)
        self.grad = self.grad + (out.data * out.grad)
    end
    out._backward = _backward
    return out
end

--- implement the tanh function for the Value class
function Value:tanh()
    local x = self.data
    local t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    local out = Value(t, { self }, 'tanh')
    local _backward = function()
        self.grad = self.grad + ((1 - t * t) * out.grad)
    end
    out._backward = _backward
    return out
end

--- implement the backpropagation for the Value
function Value:backward()
    local topo = {}
    local visited = Set.empty()

    local function build_topo(v)
        if not visited:contains(v) then
            visited:add(v)
            for _, child in ipairs(v._prev:items()) do
                build_topo(child)
            end
            table.insert(topo, v)
        end
    end

    build_topo(self)

    -- visit each node in the topological sort (in the reverse order)
    -- and call the _backward function on each Value
    self.grad = 1
    for i = #topo, 1, -1 do
        topo[i]._backward()
    end
end

-- begin test

-- local a = Value(2.0)
-- local b = Value(-3.0)
-- local c = Value(10.0)

-- local d = a * b + c
-- print(d) -- Value(data = 4.0)
-- print(d._prev)
-- print(d._op)

-- end test

-- export the Value class
return Value
