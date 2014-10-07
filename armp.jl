module ARMP

export ARMPModel, rand, randSpec, spec, ub, lb

using Optim

global const epsilon = 1e-9

# A high dimensional autoregressive model with shape parameter c and decay phi
immutable ARMPModel
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
			warn("actual c set to ", c)
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

# sample the spectrum of a model nTrial times
function randSpec(model::ARMPModel, nTrial::Integer)
	let p = model.p, n = model.n, phi = model.phi, c = model.c
		D = Array(Float64, p * nTrial)
		for kTrial in 1:nTrial
			x = rand(model)
			d, _ = eig(x * x' / n)
			D[(kTrial - 1) * p + 1:kTrial * p] = d
		end
		return D
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
	return -optimize(x -> 0.0 - f(x), bounds[1], bounds[2]).f_minimum;
end

function stieltjes(model::ARMPModel)
	global epsilon

	A(s) = 1.0 / sqrt((model.c * s + 1.0 + model.phi^2)^2 - 4.0 * model.phi^2)
	function S(z)
		s = epsilon + 1im * epsilon
		snew = 0.0
		while true
			snew = 1 / (-z + A(s))
			if abs(snew - s) < 1e-9
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

	return x -> abs(real((s(x + epsilon * 1im) - s(x - epsilon * 1im)) / 2.0im / pi))
end

end
