--- gviz.lua: A simple graphviz dot file generator
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local class = require 'lib/middleclass'
local Set = require 'util/set'

--- Declare the Node class
Node = class('Node')

--- constructor
function Node:initialize(name)
    self.name = name
end

--- override the equality operator
function Node:__eq(other)
    return self.name == other.name
end

--- tostring metamethod
function Node:__tostring()
    return self.name
end

--- Declare the Edge class
Edge = class('Edge')

--- constructor
function Edge:initialize(from, to)
    self.from = from
    self.to = to
end

--- tostring metamethod
function Edge:__tostring()
    return self.from .. ' -> ' .. self.to
end

--- override the equality operator
function Edge:__eq(other)
    return self.from == other.from and self.to == other.to
end

--- Declare the class Gviz
Gviz = class('Gviz')

--- constructor
function Gviz:initialize()
    self.nodes = Set.empty()
    self.edges = Set.empty()
end

--- add a node to the graph
function Gviz:add_node(node)
    self.nodes:add(Node(node))
end

--- add an edge to the graph
function Gviz:add_edge(from , to)
    self.edges:add(Edge(from, to))
end

--- generate the dot file
function Gviz:generate_dot()
    local dot = 'digraph G {\n'
    for _, node in ipairs(self.nodes:items()) do
        dot = dot .. '  ' .. tostring(node) .. ';\n'
    end
    for _, edge in ipairs(self.edges:items()) do
        dot = dot .. '  ' .. tostring(edge) .. ';\n'
    end
    dot = dot .. '}'
    return dot
end

--- generate png from the dot
function Gviz:generate_png(filename)
    local dot = self:generate_dot()
    local f = io.open('tmp.dot', 'w')
    if f == nil then
        error('Could not open file tmp.dot! Do you have write permissions?')
        return
    end
    f:write(dot)
    f:close()
    os.execute('dot -Tpng tmp.dot -o ' .. filename)
    os.remove('tmp.dot')
end

--- generate svg from the dot
function Gviz:generate_svg(filename)
    local dot = self:generate_dot()
    local f = io.open('tmp.dot', 'w')
    if f == nil then
        error('Could not open file tmp.dot! Do you have write permissions?')
        return
    end
    f:write(dot)
    f:close()
    os.execute('dot -Tsvg tmp.dot -o ' .. filename)
    os.remove('tmp.dot')
end

-- export the Gviz class
return Gviz

-- end of gviz.lua