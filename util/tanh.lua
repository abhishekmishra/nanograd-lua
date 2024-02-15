--- tanh.lua: implement the hyperbolic tanh function
--
-- date: 16/2/2024
-- author: Abhishek Mishra

--- tanh
-- see https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
local function tanh(x)
	return (math.exp(2 * x) - 1)/(math.exp(2 * x) + 1)
end

return tanh
