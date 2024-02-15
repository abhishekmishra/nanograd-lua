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
    self._op = _op or ''
    self.label = label or ''
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
