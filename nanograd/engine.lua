--- engine.lua: Value class for the nanograd library, based on micrograd.
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local class = require 'lib/middleclass'
local Set = require 'util/set'

--- Declare the class Value
Value = class('Value')

--- constructor
function Value:initialize(data, _children, _op)
    self.data = data
    self._op = _op or ''
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
    return Value(self.data + other.data, { self, other }, '+')
end

--- multiply this Value object with another
-- using metamethod _mul
function Value:__mul(other)
    return Value(self.data * other.data, { self, other }, '*')
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
