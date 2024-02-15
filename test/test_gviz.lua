--- test_gviz.lua: test the Gviz class
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local gviz = require 'util/gviz'

-- begin test

local g = gviz.Graph()
local a = g:add_node('a')
local b = g:add_node(gviz.Node('b', { shape = 'record', label = '{b | 10.0}' }))
local c = g:add_node(gviz.Node('c', { shape = 'diamond', color = 'red' }))

-- set node properties later
a.color = 'blue'

-- create an edge using a lable or a node object
g:add_edge('a', b)

-- create an edge using labels only
g:add_edge('b', 'c')

print(g:generate_dot())
g:generate_png('test/test_gviz.png')
g:generate_svg('test/test_gviz.svg')

-- end test