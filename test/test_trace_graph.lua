--- test_trace_graph.lua: tests for trace_graph.lua
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local trace_graph = require("util/trace_graph")
local Value = require("nanograd/engine")

-- begin test
local a = Value(2.0)
local b = Value(-3.0)
local c = Value(10.0)

local d = a * b + c

local g = trace_graph.draw_dot(d)
print(g:generate_dot())
trace_graph.draw_dot_png(d, "test/test_trace_graph.png")
trace_graph.draw_dot_svg(d, "test/test_trace_graph.svg")