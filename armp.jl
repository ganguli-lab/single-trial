export ARMPModel, rand, spec, ub, lb

using Optim, Cubature

# A high dimensional autoregressive model with shape parameter c and decay phi
immutable ARMPModel <: HDModel
	p::Integer
	n::Integer
	c::Float64
	phi::Float64

	function ARMPModel(p::Integer, n::Integer, phi::Float64)
		c = (p / n)::Float64
		if !(0 < c && 0 < phi < 1)
			error("c or phi out of bound")
		else
			return new(p, n, c, phi)
		end
	end

	function ARMPModel(cp::Float64, phi::Float64)
		if !(0 < cp && 0 < phi < 1)
			error("c or phi out of bound")
		end
		if cp < 1
			p = 1000::Int64
			n = int(p / cp)
		else
			n = 1000::Int64
			p = int(n * cp)
		end
		c = p / n
		if abs(c - cp) > 1e-6
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

function zFunc(model::ARMPModel)
	return s -> (-1 ./ s + 1 ./ sqrt((model.c * s + 1 + model.phi^2).^2 - 4 * model.phi^2))
end

# Upper bound of the eigenvalue spectrum
function ub(model::ARMPModel)
	f = zFunc(model)
	bounds = [-1 / model.c * (1 - model.phi)^2 + 1e-9, 0 - 1e-9]
	return optimize(f, bounds[1], bounds[2]).f_minimum
end

# Lower bound of the eigenvalue spectrum
function lb(model::ARMPModel)
	f = zFunc(model)
	if model.c >= 1.0
		bounds = [-eps(Float64), -1 / model.c * (1 + model.phi)^2 - eps(Float64)];
	else
		bounds = [eps(Float64), 1e16];
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
		s::Complex128 = epsilon + 1im * epsilon
		snew::Complex128 = 0.0
		while true
			snew = 1 / (-z + 1.0 / sqrt((model.c * s + 1.0 + model.phi^2)^2 - 4.0 * model.phi^2))
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
