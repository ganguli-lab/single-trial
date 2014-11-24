using Optim

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

# sample a model
function rand(model::ARMPModel)
	let n = model.n, p = model.p, phi = model.phi, sigma = model.sigma c = model.c
		x = zeros(n, p)
		P = eye(n) ./ sqrt(1.0 - phi^2)
		x[:, 1] = P * randn(n);
		for t in 2:p
			x[:, t] = phi * x[:, t - 1] + randn(n)
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
