export ARMPModel, rand, spec, ub, lb

using Optim

# A high dimensional autoregressive model with shape parameter c and decay phi
# The data dimensionality is p-by-n with correlations running across rows
immutable ARMPModel <: NoiseModel
	p::Integer
	n::Integer
	c::Number # c = p / n
	phi::Number

	function ARMPModel(p::Integer, n::Integer, phi::Number)
		c = (p / n)
		if !(0 < c && 0 < phi < 1)
			error("c or phi out of bound")
		else
			return new(p, n, c, phi)
		end
	end

	function ARMPModel(cp::Number, phi::Number)
		if !(0 < cp && 0 < phi < 1)
			error("c or phi out of bound")
		end
		if cp < 1
			p = 1000
			n = int(p / cp)
		else
			n = 1000
			p = int(n * cp)
		end
		c = p / n
		if abs(c - cp) > sqrt(eps(Float64))
			warn(string("actual c set to ", c))
		end
		return new(p, n, c, phi)
	end
end

# sample a model
function rand(model::ARMPModel)
	let p = model.p, n = model.n, phi = model.phi, c = model.c
		x = zeros(p, n)
		P = eye(p) ./ sqrt(1.0 - phi^2)
		x[:, 1] = P * randn(p);
		for t in 2:n
			x[:, t] = phi * x[:, t - 1] + randn(p)
		end
		return x
	end
end

# TODO: bit a of hack here for the c > 1 case, no reference
function zFunc(model::ARMPModel)
	# return s -> (-1 ./ s + 1 ./ sqrt((model.c * s + 1 + model.phi^2).^2 - 4 * model.phi^2))
	return s -> (-1 ./ s + sign(s + (1 + model.phi^2) / model.c) ./ sqrt((model.c * s + 1 + model.phi^2).^2 - 4 * model.phi^2))
end

# Upper bound of the eigenvalue spectrum
function ub(model::ARMPModel)
	f = zFunc(model)
	bounds = [-1 / model.c * (1 - model.phi)^2 + sqrt(eps(Float64)), -sqrt(eps(Float64))]
	return optimize(f, bounds[1], bounds[2]).f_minimum
end

# Lower bound of the eigenvalue spectrum
function lb(model::ARMPModel)
	f = zFunc(model)
	if model.c >= 1.0
		bounds = [-1e16, -1 / model.c * (1 + model.phi)^2 - sqrt(eps(Float64))];
	else
		bounds = [sqrt(eps(Float64)), 1e16];
	end
	if bounds[1] >= bounds[2]
		println(model.p, " ", model.n)
	end
	return -optimize(x -> 0.0 - f(x), bounds[1], bounds[2]).f_minimum;
end

function stieltjes(model::ARMPModel)
	global epsilon

	function S(z)
		global zPoints
		s::Complex128 = 0.1 + 1im
		snew::Complex128 = 0.0
		while true
			snew = 1 / (-z + 1.0 / sqrt((model.c * s + 1.0 + model.phi^2)^2 - 4.0 * model.phi^2))
			# println(snew)
			if abs(snew - s) < epsilon
				return s
			end
			s = snew
		end
	end
	return S
end

# Returns a spectrum function for the given model
function spec(model::ARMPModel)
	global epsilon
	s = stieltjes(model)

	#return x -> abs(real((s(x + epsilon * 1im) - s(x - epsilon * 1im)) / 2.0im / pi))
	return x -> abs(imag(s(x + epsilon * 1im))) / pi
end