--- Graph.lua: A simple graphviz dot file generator
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local class = require 'lib/middleclass'
local Set = require 'util/set'

--- Declare the Node class
Node = class('Node')

--- constructor
function Node:initialize(node_obj, config)
    self.node = node_obj
    self.name = tostring(self.node)
    self.config = config or {}
    self.shape = self.config.shape or 'circle'
    self.color = self.config.color or 'black'
    self.style = self.config.style or 'solid'
    self.label = self.config.label or self.name
    self.fontname = self.config.fontname or 'Arial'
    self.fontsize = self.config.fontsize or 12
end

--- override the equality operator
function Node:__eq(other)
    return self.node == other.node
end

--- tostring metamethod
function Node:__tostring()
    return '"' .. self.name .. '"'
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
    return '"' .. tostring(self.from) .. '" -> "' .. tostring(self.to) .. '"'
end

--- override the equality operator
function Edge:__eq(other)
    return self.from == other.from and self.to == other.to
end

--- Declare the class Graph
Graph = class('Graph')

--- constructor
function Graph:initialize()
    self.nodes = Set.empty()
    self.edges = Set.empty()
end

--- add a node to the graph
function Graph:add_node(node)
    local node = node or 'empty'
    local node_obj = nil
    -- if type of node is Node, then add it directly
    -- check if node has an attribute class and is an instance of Node
    if type(node) == 'table' and node.class == Node then
        node_obj = node
        self.nodes:add(node)
    else
        node_obj = Node(node)
        self.nodes:add(node_obj)
    end
    return node_obj
end

--- add an edge to the graph
function Graph:add_edge(from, to)
    self.edges:add(Edge(from, to))
end

--- generate the dot file
function Graph:generate_dot(rankdir)
    local rankdir = rankdir or 'LR'
    local dot = 'digraph G {\n'
    dot = dot .. '  rankdir=' .. rankdir .. ';\n'
    for _, node in ipairs(self.nodes:items()) do
        -- show the node based on its properties
        dot = dot .. '  ' .. tostring(node)
            .. ' [shape='
            .. node.shape .. ', color='
            .. node.color .. ', style='
            .. node.style .. ', label="'
            .. node.label .. '", fontname="'
            .. node.fontname .. '", fontsize='
            .. node.fontsize
            .. '];\n'
    end
    for _, edge in ipairs(self.edges:items()) do
        dot = dot .. '  ' .. tostring(edge) .. ';\n'
    end
    dot = dot .. '}'
    return dot
end

--- generate png from the dot
function Graph:generate_png(filename)
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
function Graph:generate_svg(filename)
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

-- export the classes
return { Graph = Graph, Node = Node, Edge = Edge}

-- end of Graph.lua
