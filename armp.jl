module ARMP

export ARMPModel, spectrum, ub, lb

using Optim

global epsilon = 1e-9

# A high dimensional autoregressive model with shape parameter c and decay phi
immutable ARMPModel
	c::Float64
	phi::Float64
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
		s = 0.1 + 1im * epsilon
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
function spectrum(model::ARMPModel)
	global epsilon
	s = stieltjes(model)

	return x -> real((s(x + epsilon * 1im) - s(x - epsilon * 1im)) / 2.0im / pi)
end

end
