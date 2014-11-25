# using Optim

# A high dimensional AR noisemodel with shape parameter c, decay phi
# and input standard deviation sigma
# The data dimensionality is n-by-p with correlations running across rows
immutable ARMPModel <: NoiseModel
	n::Integer # number of neurons
	p::Integer # number of trials
	c::Number # shape = n / p
	phi::Number
	sigma::Number
end

# Constructors

function ARMPModel(n::Integer, p::Integer, phi::Number, sigma::Number)
	c = n / p
	if 0 >= c || 0 >= sigma || 0 >= phi || 1 <= phi
		error("c, sigma, or phi out of bound")
	else
		return new(n, p, c, phi, sigma)
	end
end

function ARMPModel(cp::Number, phi::Number, sigma::Number)
	if 0 >= cp || 0 >= sigma || 0 >= phi || 1 <= phi
		error("c, sigma, or phi out of bound")
	end
	if cp < 1
		n = 1000
		p = int(n / cp)
	else
		p = 1000
		n = int(p * cp)
	end
	c = p / n
	if abs(c - cp) > sqrt(eps(Float64))
		warn(string("actual c set to ", c))
	end
	return new(n, p, c, phi, sigma)
end

## Sampling

function rand(m::ARMPModel)
	let n = m.n, p = m.p, phi = m.phi, sigma = m.sigma c = m.c
		x = zeros(n, p)
		P = eye(n) ./ sqrt(1.0 - phi^2)
		x[:, 1] = P * randn(n);
		for t in 2:p
			x[:, t] = phi * x[:, t - 1] + randn(n)
		end
		return x
	end
end

randfull(m::ARMPModel) = {:Z => rand(m)}

## Eigenvalue and singular value spectrum
##   J. Yao, 2011

# TODO: bit a of hack here for the c > 1 case, no reference
zFunc(m::ARMPModel, s) = (-1 ./ s + sign(s + (1 + m.phi^2) / m.c) ./ sqrt((m.c * s + 1 + m.phi^2).^2 - 4 * m.phi^2))

# Upper bound of the eigenvalue spectrum
function ev_ub(m::ARMPModel)
	f = zFunc(m)
	bounds = [-1 / m.c * (1 - m.phi)^2 + sqrt(eps(Float64)), -sqrt(eps(Float64))]
	return optimize(f, bounds[1], bounds[2]).f_minimum
end

function ev_lb(m::ARMPModel)
	f = zFunc(m)
	if m.c >= 1.0
		bounds = [-1e16, -1 / m.c * (1 + m.phi)^2 - sqrt(eps(Float64))];
	else
		bounds = [sqrt(eps(Float64)), 1e16];
	end
	if bounds[1] >= bounds[2]
		println(m.p, " ", m.n)
	end
	return -optimize(x -> 0.0 - f(x), bounds[1], bounds[2]).f_minimum;
end

sv_lb(m::ARMPModel) = sqrt(ev_lb(m))

sv_ub(m::ARMPModel) = sqrt(ev_ub(m))

# function stieltjes(m::ARMPModel)
# 	global epsilon
#
# 	function S(z)
# 		global zPoints
# 		s::Complex128 = 0.1 + 1im
# 		snew::Complex128 = 0.0
# 		while true
# 			snew = 1 / (-z + 1.0 / sqrt((m.c * s + 1.0 + m.phi^2)^2 - 4.0 * m.phi^2))
# 			# println(snew)
# 			if abs(snew - s) < epsilon
# 				return s
# 			end
# 			s = snew
# 		end
# 	end
# 	return S
# end
#
# # Returns a spectrum function for the given model
# function spec(model::ARMPModel)
# 	global epsilon
# 	s = stieltjes(model)
#
# 	#return x -> abs(real((s(x + epsilon * 1im) - s(x - epsilon * 1im)) / 2.0im / pi))
# 	return x -> abs(imag(s(x + epsilon * 1im))) / pi
# end
