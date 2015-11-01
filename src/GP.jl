module GP

using Optim

export Kernel
export val, deriv, fs, tc, tc_interp, tc_interp_uniform
export cov, deriv_cov
export deriv_cov_uniform, cov_uniform, inv_cov_uniform
export e_cos_subopt, e_cos_large

immutable Kernel
	Delta::Number
end

# kernel value
function val(theta, ker::Kernel)
    kappa = exp(-ker.Delta)
    rst = (1 - kappa)^2 ./ (1 - 2kappa * cos(theta) + kappa^2)
    rst[cos(theta) .== 1] = 1.0
    if any(isnan(rst)) || any(isinf(rst))
        throw("invalid value in kernel val")
    end
    return rst
end

# kernel derivative
function deriv(theta, ker::Kernel)
	kappa = exp(-ker.Delta)
	rst = -2kappa * (1 - kappa)^2 ./ (1 - 2kappa*cos(theta) + kappa^2).^2 .* sin(theta)
    rst[cos(theta) .== 1] = 0.0
    if any(isnan(rst)) || any(isinf(rst))
        throw("invalid value in kernel deriv")
    end
    return rst
end

# kernel derivative to shape parameter
function deriv_delta(theta, ker::Kernel)
    rst = -2 * sin(theta / 2).^2 * sinh(1 * ker.Delta) ./ (cos(theta) / ker.Delta - cosh(1 * ker.Delta) / ker.Delta).^2
    if isinf(cosh(1 * ker.Delta))
        rst[:] = 0.0
    end
    rst[cos(theta) .== 1] = 0.0
    if any(isnan(rst)) || any(isinf(rst))
        throw("invalid value in kernel delta derivative")
    end
    return rst
end

#### Basic stuff ####
# kernel fourier series
function fs(k, ker::Kernel)
    2pi * tanh(1 / 2 * ker.Delta) * exp(-abs(k) * ker.Delta)
end

# arbituary covariancesdd
function cov(theta, ker::Kernel)
    return val(theta .- theta', ker)
end

function deriv_cov(theta, ker::Kernel)
    return deriv_delta(theta .- theta', ker)
end

# uniform covariances
function cov_uniform(C, ker::Kernel)
    θs = linspace(0, 2π, C+1)[1:end-1]
    row = val(θs, ker)
    hcat(map(x -> circshift(row, x), 0:(C-1))...)'
end

function deriv_cov_uniform(C, ker::Kernel)
    θs = linspace(0, 2π, C+1)[1:end-1]
    row = deriv_delta(θs, ker)
    hcat(map(x -> circshift(row, x), 0:(C-1))...)'
end

function inv_cov_uniform(C, ker::Kernel; σn=0.0)
    u, evs = U(C), Float64[lc(k, C, ker) + σn^2 for k in 0:(C-1)]
    return real(u * diagm(1 ./ evs) * u')
end

# generate tuning curves
function tc(res, ntc, ker::Kernel)
    u, z = U(res), randn(res, ntc)
    ls = convert(Array{Float64, 1}, [sqrt(lc(k, res, ker)) for k in 0:(res - 1)])
    x = real(u * diagm(ls) * u' * z)
    return linspace(0, 2π, res+1)[1:res], x
end

function tc2(θs, ntc, ker::Kernel; σn=1e-16)
    T = length(θs)
    z = randn(T, ntc)
    Csqrt = sqrtm(cov(θs, ker) + σn^2 * eye(T))
    return real(Csqrt * z)
end

# interpolation the rest of the tuning curves
function tc_interp_uniform(xs, θs_new, ker::Kernel; Cn=nothing)
    ncond, _ = size(xs)
    θs = linspace(0, 2π, ncond + 1)[1:end-1]

    P = val(θs_new .- θs', ker)
    C = cov_uniform(ncond, ker)
    if Cn != nothing; C = C + Cn; end
    Cinv = inv(C + eye(ncond) * 1e-9)

    xs_new = P * Cinv * xs
    σs_new = sqrt(abs(diag(val(θs_new .- θs_new', ker) - P * Cinv * P')))
    return xs_new, σs_new
end

# interpolation with arbitrary training data and test stimulus
function tc_interp(θs, xs, θs_new, ker::Kernel; Cn=nothing)
    P = val(θs_new .- θs', ker)
    C = val(θs .- θs', ker)
    if Cn != nothing
        C = C + diagm(diag(Cn))
    end
    Cinv = real(inv(C))

    xs_new = P * Cinv * xs
    σs_new = sqrt(abs(diag(val(θs_new .- θs_new', ker) - P * Cinv * P')))
    return xs_new, σs_new
end

#### Inference ####
function decode(rs_test, rs_train, Δ, σs, σn; res=1000)
    obj(θ) = decode_scr(θ, rs_test, rs_train, Δ, σs, σn)

    θtest = linspace(0, 2π, res+1)[1:end-1]
    nll_test = map(obj, θtest)

    _, ix = findmin(nll_test)
    return θtest[ix]
end

function decode_scr(θ, rs_test, rs_train, Δ, σs, σn)
    nnrn, ntrial = size(rs_test)
    ncond, nnrn, ntrial = size(rs_train)
    rs_train_μ = squeeze(mean(rs_train, 3), 3)
    rs_test_μ = mean(rs_test)
    ker = GP.Kernel(Δ)

    θtrains = linspace(0, 2π, ncond + 1)[1:end - 1]
    Cx = σs^2 * cov_uniform(ncond, ker) + eye(ncond) * σn^2 / ntrial
    Cy = σs^2 + σn^2
    P = val(θ - θtrains, ker)' * σs^2
    μ = (P * inv(Cx) * rs_train_μ)[1]

    CC = Cy - (P * inv(Cx) * P')[1]

    scr = 0.0
    for nix in 1:nnrn
        scr += log(CC)
        scr += 0.5 * (rs_test_μ - μ)^2
    end
    return scr
end

function nll_helper!(rs, Δ, σs, σn; gpars=nothing)
    ncond, nnrn, ntrial = size(rs)
    rs_mean = squeeze(mean(rs, 3), 3)

    ker = Kernel(Δ)
    C = cov_uniform(ncond, ker) * σs^2
    M = inv(C + σn^2 / ntrial * eye(ncond))
    Cinv = C - C * M * C

    nll = nnrn * ncond * ntrial * log(σn)
    nll += 0.5 * nnrn * logdet(ntrial / σn^2 * C + eye(ncond))
    nll += 0.5 * sum(rs.^2) / σn^2
    nll -= 0.5 * ntrial^2 / σn^4 * trace(rs_mean' * Cinv * rs_mean)

    if gpars != nothing
        CdΔ = deriv_cov_uniform(ncond, ker)

        for n in 1:nnrn
            v = rs_mean[:, n]

            # Δ gradient
            gpars[1] += 0.5 * trace(M * CdΔ)
            gpars[1] -= 0.5 * (v' * CdΔ * v)[1] * ntrial^2 / σn^4
            gpars[1] += (v' * CdΔ * M * C * v)[1] * ntrial^2 / σn^4
            gpars[1] -= 0.5 * (v' * C * M * CdΔ * M * C * v)[1] * ntrial^2 / σn^4

            # σs gradient
            gpars[2] += trace(M * C) / σs
            gpars[2] -= (v' * C * v)[1] * ntrial^2 / σn^4 / σs
            gpars[2] += (v' * C * M * C * v)[1] * ntrial^2 / σn^4 / σs * 2
            gpars[2] -= (v' * C * M * C * M * C * v)[1] * ntrial^2 / σn^4 / σs

            # σn gradient
            gpars[3] += ncond * ntrial / σn
            gpars[3] -= trace(M * C) / σn
            gpars[3] -= sum(rs[:, n, :].^2 / σn^3)
            tmp = 2 * ntrial^2 / σn^5 * C - 2 * ntrial^2 / σn^5 * C * M * C
            tmp -= ntrial / σn^3 * C * M * M * C
            gpars[3] += (v' * tmp * v)[1]
        end
    end

    return nll
end

function infer(rs; Δ0=0.3, Δmin=1e-6, σsmin=1e-6, σnmin=1e-6)
    function cost(z)
        let Δ = z[1]^2 + Δmin, σs = z[2]^2 + σsmin, σn = z[3]^2 + σnmin
            return nll_helper!(rs, Δ, σs, σn)
        end
    end
    function cost_g!(z, g::Vector; return_cost=false)
        g[:] = 0
        let Δ = z[1]^2 + Δmin, σs = z[2]^2 + σsmin, σn = z[3]^2 + σnmin
            E = nll_helper!(rs, Δ, σs, σn; gpars=g)
            g[:] = g[:] .* (2 * z)

            if return_cost; return E; end
        end
    end
    cost_and_g!(z, g::Vector) = cost_g!(z, g; return_cost=true)

    σs0, σn0 = sqrt(var(mean(rs, 3)[:])), sqrt(mean(var(rs, 3)))
    z0 = [sqrt(Δ0 - Δmin), sqrt(σs0 - σsmin), sqrt(σn0 - σnmin)]
    #=rst = Optim.optimize(DifferentiableFunction(cost, cost_g!, cost_and_g!), z0, method=:cg, ftol=1e-9).minimum=#
    rst = Optim.optimize(cost, z0, method=:nelder_mead, ftol=1e-9)
    sol = rst.minimum

    return sol[1]^2 + Δmin, sol[2]^2 + σsmin, sol[3]^2 + σnmin, rst.f_minimum
end

#### Circulant matrix stuff used in GP posterior ####
# eigenvectors for circulant matrices
u_buffer = Dict{Tuple{Integer, Integer}, Array{Complex64, 1}}()
function u(k::Integer, T::Integer)
    if haskey(u_buffer, (k, T)); return u_buffer[(k, T)]; end
    μ = 0:(T-1)
    rst = 1 / sqrt(T) * exp(1im * k * μ * 2π / T)
    u_buffer[(k, T)] = rst
    return rst
end

U(T) = hcat([u(k, T) for k in 0:(T-1)]...)

# eigenvalues for the observation covariance matrix
function lc(k, T, ker::Kernel)
    d = ker.Delta
    if 1 - exp(-T * d) == 0
        l = 1.0
    else
	    l = T * tanh(1 / 2 * d) * (exp(-(T-k)*d) + exp(-k*d)) / (1 - exp(-T*d))
    end
    if isinf(l) || isnan(l) || l < 0
        throw("eigenvalue is invalid")
    end
    return l
end

# the C matrix, or the covariance of observations
function C(T, ker::Kernel)
    rst = 0
    for k in 0:(T - 1)
        rst += u(k, T) * u(k, T)' * lc(k, T, ker)
    end
    return real(rst)
end

# eigenvalues for the posterior tuning curve correlation matrix
function lq(k, T, ker::Kernel; ε=1e-12)
    d = ker.Delta
	return T * tanh(d / 2)^2 * (exp(-(2T-2k)*d) + exp(-2k*d)) / (1 - exp(-2T*d)) + ε
end

# the Q matrix, or the expected correlation of the posterior tuning curves
function Q(T, ker::Kernel)
    rst = 0
    for k in 0:(T - 1)
        rst += real(u(k, T) * u(k, T)' * lq(k, T, ker))
    end
    return rst
end

# Oracle cosine suboptimal
function e_cos_subopt(N, sigman, ker::Kernel)
    k1 = fs(1, ker)
    denom = (N + 1) * k1  + 2pi * (1 + sigman^2)
    0.5 * (1 - N * k1 / denom)
end

# Oracle cosine large N
function e_cos_large(N, sigman, ker::Kernel)
    k1 = fs(1, ker)
    denom = N * k1  + 2pi * sigman^2
    0.5 * (1 - N * k1 / denom)
end
end
