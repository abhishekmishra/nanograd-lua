--- trace_graph.lua: Draw the graph for the Value object.
--
-- Date: 15/02/2024
-- Author: Abhishek Mishra

local set = require 'util/set'
local gviz = require 'util/gviz'

--- trace the graph for the Value object
-- and return the nodes and edges as a set
local function trace(root)
    local nodes = set.empty()
    local edges = set.empty()

    local function visit(node)
        if not nodes:contains(node) then
            nodes:add(node)
            for _, child in ipairs(node._prev:items()) do
                edges:add({ child, node })
                visit(child)
            end
        end
    end

    visit(root)
    return nodes, edges
end

--- draw the dot graph for the Value object
local function draw_dot(root)
    local nodes, edges = trace(root)

    -- create new graph using gviz.Graph
    local g = gviz.Graph()

    for _, node in ipairs(nodes:items()) do
        -- for any value in the graph, create a record node
        g:add_node(Node(node.id, {
            shape = 'record',
            label = '{ ' .. node.label .. ' | data ' .. node.data .. '}'
        }))

        -- if this value is a result of some operation, create an op node for it
        if node._op ~= '' then
            g:add_node(Node(tostring(node.id) .. node._op, {
                shape = 'ellipse',
                label = node._op,
                fontsize = 18,
                color = 'blue'
            }))
            g:add_edge(tostring(node.id) .. node._op, node.id)
        end
    end

    -- connect first node to the op node of the second node in each edge
    for _, edge in ipairs(edges:items()) do
        g:add_edge(edge[1].id,
            edge[2]._op ~= '' and
            (tostring(edge[2].id) .. edge[2]._op) or edge[2].id)
    end

    return g
end

--- create a svg graph of the Value object
-- using the draw_dot function
local function draw_dot_svg(root, filename)
    local g = draw_dot(root)
    g:generate_svg(filename)
end

--- create a png graph of the Value object
-- using the draw_dot function
local function draw_dot_png(root, filename)
    local g = draw_dot(root)
    g:generate_png(filename)
end

return {
    trace = trace,
    draw_dot = draw_dot,
    draw_dot_svg = draw_dot_svg,
    draw_dot_png = draw_dot_png
}
