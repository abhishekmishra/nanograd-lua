--- test_trace_graph.lua: tests for trace_graph.lua
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local trace_graph = require("util/trace_graph")
local Value = require("nanograd/engine")

-- begin test
local a = Value(2.0)
a.label = 'a'
local b = Value(-3.0)
b.label = 'b'
local c = Value(10.0)
c.label = 'c'

local e = a * b
e.label = 'e'

local d = e + c
d.label = 'd'

local f = Value(-2.0)
f.label = 'f'

local L = d * f
L.label = 'L'

-- print(L, L._op, L._prev)

local g = trace_graph.draw_dot(L)
print(g:generate_dot())
trace_graph.draw_dot_png(L, "test/test_trace_graph.png")
trace_graph.draw_dot_svg(L, "test/test_trace_graph.svg")