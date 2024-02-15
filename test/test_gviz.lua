--- test_gviz.lua: test the Gviz class
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local Gviz = require 'util/gviz'

-- begin test

local g = Gviz()
g:add_node('a')
g:add_node('b')
g:add_node('c')
g:add_edge('a', 'b')
g:add_edge('b', 'c')
print(g:generate_dot())
g:generate_png('test/test_gviz.png')
g:generate_svg('test/test_gviz.svg')

-- end test