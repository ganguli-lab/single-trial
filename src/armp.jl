using Optim, Cubature, Polynomials

armp_epsilon = 1e-4

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
		ARMPModel(n, p, c, phi, sigma)
	end
end

function ARMPModel(cp::Number, phi::Number, sigma::Number)
	if 0 <= cp || 0 <= sigma || 0 <= phi || 1 >= phi
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
	ARMPModel(n, p, c, phi, sigma)
end

## Sampling

function rand(m::ARMPModel)
	let n = m.n, p = m.p, c = m.c, phi = m.phi, sigma = m.sigma
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
zFunc(m::ARMPModel, s::Number) = -1 ./ s + sign(s + (1 + m.phi^2) / m.c) ./ sqrt((m.c * s + 1 + m.phi^2).^2 - 4 * m.phi^2)

# Upper bound of the eigenvalue spectrum
function ev_ub(m::ARMPModel)
	bounds = [-1 / m.c * (1 - m.phi)^2 + sqrt(eps(Float64)), -sqrt(eps(Float64))]
	return m.sigma^2 * m.p * optimize(x -> zFunc(m, x), bounds[1], bounds[2]).f_minimum
end

function ev_lb(m::ARMPModel)
	if m.c >= 1.0
		bounds = [-1e16, -1 / m.c * (1 + m.phi)^2 - sqrt(eps(Float64))];
	else
		bounds = [sqrt(eps(Float64)), 1e16];
	end
	if bounds[1] >= bounds[2]
		println(m.p, " ", m.n)
	end
	return -m.sigma^2 * m.p * optimize(x -> 0.0 - zFunc(m, x), bounds[1], bounds[2]).f_minimum;
end

sv_lb(m::ARMPModel) = sqrt(ev_lb(m))

sv_ub(m::ARMPModel) = sqrt(ev_ub(m))

function stieltjes(m::ARMPModel, z::Number; max_iter=10000)
	# coefficients for the 4th order polynomial
	coefs = [(1 - m.phi^2)^2,
					 2 * (m.c * (1 + m.phi^2) + z * (1 - m.phi^2)^2),
					 (1 - m.phi^2)^2 * z^2 + 4 * m.c * z * (1 + m.phi^2) + m.c^2 - 1,
					 2 * m.c * z * (m.c + z + z * m.phi^2),
					 m.c^2 * z^2]
	candidates = roots(Poly(coefs))
	# check against original equation
	tmp = map(s -> abs(z + 1 / s - 1 / sqrt((m.c * s + 1 + m.phi^2)^2 - 4 * m.phi^2)), candidates)
	tmp = tmp[tmp .< sqrt(eps(Float64))]
	# HACK, returning the value with the largest imag part
	_, ix = findmax(abs(imag(candidates)))
	return candidates[ix]
end

function ev_spec(m::ARMPModel)
	x -> abs(imag(stieltjes(m, (x / m.p / m.sigma^2) + 1e-3 * 1im))) / pi / m.p / m.sigma^2
end

function sv_spec(m::ARMPModel)
	es = ev_spec(m)
	x -> es(x^2) * 2 * x
end

## Low-rank perturbations. see:
##   Benaych-Georges, F. & Nadakuditi, R., 2012

# dtransform for the neuron by neuron covariance
function dtransform(m::ARMPModel, z::Number)
	l, u, mu = ev_lb(m), ev_ub(m), ev_spec(m)
	tmp = pquadrature(t -> sqrt(z) / (z - t) * mu(t), l + armp_epsilon, u - armp_epsilon; reltol=1e-3, abstol=1e-3)[1]
	if m.n > m.p # do the math for data matrix transposed
		tmp *= m.c
		tmp * (tmp / m.c + (1 - 1 / m.c) / sqrt(z))
	else
		tmp * (m.c * tmp + (1 - m.c) / sqrt(z))
	end
end

ev_infloor(m::ARMPModel) = 1 / dtransform(m, ev_ub(m) + armp_epsilon)

sv_infloor(m::ARMPModel) = sqrt(ev_infloor(m))

sv_outfloor(m::ARMPModel) = sv_ub(m)

ev_outfloor(m::ARMPModel) = ev_ub(m)

## Special model to treat the ARMP Correlation matrix
immutable ARMPCorrModel
	armp::ARMPModel
end

function dtransform(m::ARMPCorrModel, z::Number)
	# singular/eigen values for the correlation
	l, u, mu = ev_lb(m.armp), ev_ub(m.armp), ev_spec(m.armp)
	tmp = pquadrature(t -> z / (z^2 - t^2) * mu(t), l + armp_epsilon, u - armp_epsilon; reltol=1e-3, abstol=1e-3)[1]
	if m.armp.n > m.armp.p; tmp *= m.armp.c; end
	tmp * tmp
end

ev_infloor(m::ARMPCorrModel) = 1 / sqrt(dtransform(m, ev_ub(m.armp) + armp_epsilon))
