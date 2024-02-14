-- set.lua: simple set implementation using middleclass
-- based on the implementation of set in Programming in Lua
-- by Roberto Ierusalimschy
-- see http://www.lua.org/pil/11.5.html
--
-- Author: Abhishek Mishra
-- Date: 14/02/2024

local class = require 'middleclass'

-- Declare the class Set
Set = class('Set')

-- constructor
function Set:initialize(list)
    self.values = {}
    for _, l in ipairs(list) do
        self.values[l] = true
    end
end

-- empty set
function Set.empty()
    return Set({})
end

-- clear the set
function Set:clear()
    self.values = {}
end

-- add an element to the set
function Set:add(elem)
    self.values[elem] = true
end

-- remove an element from the set
function Set:remove(elem)
    self.values[elem] = nil
end

-- check if an element is in the set
function Set:contains(elem)
    return self.values[elem] == true
end

-- return the union of two sets
function Set:union(other)
    local res = Set(self:items())
    for k, _ in pairs(other.values) do
        res.values[k] = true
    end
    return res
end

-- return the intersection of two sets
function Set:intersection(other)
    local res = Set({})
    for k, _ in pairs(self.values) do
        res.values[k] = other.values[k]
    end
    return res
end

-- elements of the set
function Set:items()
    local res = {}
    for k, _ in pairs(self.values) do
        table.insert(res, k)
    end
    return res
end

-- string representation of the set
function Set:__tostring()
    return "{" .. table.concat(self:items(), ", ") .. "}"
end

-- export the Set class
return Set

-- some test code
-- local s1 = Set({1, 2, 3})
-- print(s1)

-- s1:add(4)
-- print(s1)

-- s1:remove(2)
-- print(s1)

-- print(tostring(s1) .. ' contains 3? ' .. tostring(s1:contains(3)))
-- print(tostring(s1) .. ' contains 0? ' .. tostring(s1:contains(0)))

-- local s2 = Set({3, 4, 5})
-- print(s1:union(s2))

-- print(s1:intersection(s2))