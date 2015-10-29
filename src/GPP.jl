module GPP

export inference_uniform

using GP, Distributions, Optim, Debug, NPZ, ArrayViews, Roots

function decode(rs, xs, σs, κ, μ, Δ)
    obj(θ) = decode_nll(θ, rs, xs, σs, κ, μ, Δ)
    return (Optim.optimize(obj, -2π, 2π).minimum + 2π) % 2π
end

function decode_nll(θ, rs, xs, σs, κ, μ, Δ)
    nnrn, ntrial = size(rs)
    ncond, nnrn = size(xs)
    ker = GP.Kernel(Δ)

    θtrains = linspace(0, 2π, ncond + 1)[1:end-1]

    C, P = GP.cov_uniform(ncond, ker), GP.val(θ - θtrains, ker)'

    nll = 0.0
    for n in 1:nnrn
        Cinv = inv(C + diagm(σs[:, n].^2))
        # conditional mean and variance
        m, s = (P * Cinv * xs[:, n])[1], sqrt(1 - P * Cinv * P')[1]
        # poisson minima
        xp = log(mean(rs[n, :]) / μ) / κ + κ / 2
        # poisson hessian
        h = κ^2 * sum(rs[n, :])

        nll +=  0.5 * (log(1 + s^2 * h) + (xp - m)^2 / (1 / h + s^2)^2)
    end
    return nll
end

function infer_joint(rs; Δ0=0.5, κ0=0.5, Δmin=0.1, μmin=0.1)
    # rs: ncond by nnrn by ntrial
    ncond, nnrn, ntrial = size(rs)

    # cost function and associated gradient
    function cost(z)
        # recover the variables
        κ, μ, Δ, xs = z[1], z[2]^2 + μmin, z[3]^2 + Δmin, reshape(z[4:end], ncond, nnrn)
        # compute the cost
        return nll_helper!(κ, μ, Δ, xs, rs)
    end
    function cost_g!(z, g::Vector; return_cost=false)
        # recover the variables
        κ, μ, Δ, xs = z[1], z[2]^2 + μmin, z[3]^2 + Δmin, reshape(z[4:end], ncond, nnrn)
        # calc grad
        E = nll_helper!(κ, μ, Δ, xs, rs; gpars=view(g, 1:3), gxs=view(g, 4:length(g)))
        g[1:3] = g[1:3] .* [1, 2 * z[2], 2 * z[3]]
        # cap gradients
        mask = abs(g) .> 1e9
        if any(mask)
            g[mask] = 1e9 * sign(g[mask])
        end

        if any(isinf(g))
            throw("poop")
        end

        if return_cost; return E; end
    end
    cost_and_g!(z, g::Vector) = cost_g!(z, g; return_cost=true)

    # reasonable initial guesses for the averaged firing rate
    μ0 = mean(rs[:])
	# initialize tuning curve, ncond by nnrn
    xs0 = log(squeeze(mean(rs, 3), 3) / μ0) / κ0 + κ0 / 2
    # initial vector
    z0 = vcat([κ0, sqrt(μ0 - μmin), sqrt(Δ0 - Δmin)], xs0[:])
    # solve
    rst = Optim.cg(DifferentiableFunction(cost, cost_g!, cost_and_g!), z0, grtol=1e-4, ftol=1e-6, iterations=2000, show_trace=false, extended_trace=false)
    z = rst.minimum
    # return values
    return z[1], z[2]^2 + μmin, z[3]^2 + Δmin, reshape(z[4:end], ncond, nnrn)
end

function infer_em(rs; Δ0=0.5, κ0=0.5, Δmin=0.1, μmin=0.1, maxiter=1000)
    ncond, nnrn, ntrial = size(rs)
    # reasonable initial guesses for the averaged firing rate
    μ0 = mean(rs[:])
	# initialize tuning curve, ncond by nnrn
    xs = log(squeeze(mean(rs, 3), 3) / μ0) / κ0 + κ0 / 2
    xs_new = zeros(ncond, nnrn)
    # initial vector
    pars = [κ0, sqrt(μ0 - μmin), sqrt(Δ0 - Δmin)]
    pars_new = zeros(3)

    iter = 0
    # iterations
    while true
        iter += 1
        # solve for tuning curves
        cost(z) = nll_helper!(pars[1], pars[2]^2 + μmin, pars[3]^2 + Δmin,
            reshape(z, ncond, nnrn), rs)
        function cost_g!(z, g::Vector; return_cost=false)
            nll = nll_helper!(pars[1], pars[2]^2 + μmin, pars[3]^2 + Δmin,
                reshape(z, ncond, nnrn), rs; gxs=g, gpars=nothing)
            mask = abs(g) .> 1e9
            if any(mask); g[mask] = 1e9 * sign(g[mask]); end
            if return_cost; return nll; end
        end
        cost_and_g!(z, g::Vector) = cost_g!(z, g; return_cost=true)
        xs_new = reshape(Optim.cg(DifferentiableFunction(cost, cost_g!, cost_and_g!), xs[:], grtol=1e-4, ftol=1e-6).minimum, ncond, nnrn)

        # solve for parameters
        cost(z) = nll_helper!(z[1], z[2]^2 + μmin, z[3]^2 + Δmin, xs_new, rs)
        function cost_g!(z, g::Vector; return_cost=false)
            nll = nll_helper!(z[1], z[2]^2 + μmin, z[3]^2 + Δmin,
                xs_new, rs; gxs=nothing, gpars=g)
            g[:] = g[:] .* [1, 2 * z[2], 2 * z[3]]
            if return_cost; return nll; end
        end
        cost_and_g!(z, g::Vector) = cost_g!(z, g; return_cost=true)
        pars_new = Optim.cg(DifferentiableFunction(cost, cost_g!, cost_and_g!), pars, grtol=1e-4, ftol=1e-6).minimum

        # stopping condition
        if maximum(abs(pars_new - pars) ./ pars) < 1e-3
            if maximum(mapslices(norm, xs_new - xs, 1) ./ mapslices(norm, xs, 1)) < 1e-3
                break
            end
        end
        if iter > maxiter
            warn("Max iteration reached")
            break
        end

        xs, pars = copy(xs_new), copy(pars_new)
    end

    return pars[1], pars[2]^2 + μmin, pars[3]^2 + Δmin, xs
end

# negative loglikelihood energy function of the GPP model
function nll_helper!(κ, μ, Δ, xs, rs; gpars=nothing, gxs=nothing)
    # dimensions
    ncond, nnrn, ntrial = size(rs)

    # covariance and associated functions
    ker = Kernel(Δ)
    C = cov_uniform(ncond, ker)
    Cinv = inv(C + eye(ncond) * 1e-12)
    Ceigs = [GP.lc(c, ncond, ker) + 1e-12 for c in 0:(ncond-1)]
    if any(Ceigs .<= 0); throw("Non-negative covariance eigenvalues"); end
    Clogdet = sum(map(log, Ceigs))
    if isnan(Clogdet) || isinf(Clogdet); throw("Invalid log determinant value"); end
    CdΔ = GP.deriv_cov_uniform(ncond, ker)
    if any(isinf(CdΔ)) || any(isnan(CdΔ)); throw("Covariance derivative invalid"); end

    # firing rates
    λs = μ * exp(κ * (xs - κ / 2))
    if any(isinf(λs))
        #=throw("Infinite firing rates")=#
        λs[isinf(λs)] = 1e9
    end
    logλs = log(μ) + κ * (xs - κ / 2)

    # negative log likelihood
    nll = 0.5 * Clogdet * nnrn + 0.5 * sum(xs * xs' .* Cinv)
    if isnan(nll) || isinf(nll); throw("Gaussian likelihood invalid: $nll"); end
    nll += sum(λs .- rs .* logλs)
    if isinf(nll); nll = 1e9 * sign(nll); end
    if isnan(nll) || isinf(nll); throw("Poisson likelihood invalid: $nll"); end

    # parameter gradients
    if gpars != nothing
        # κ
        gpars[1] = sum((λs .- rs) .* (xs - κ))
        # μ
        gpars[2] = sum(λs .- rs) / μ
        # Δ
        gpars[3] = 0.5 * trace((Cinv * nnrn - Cinv * xs * xs' * Cinv) * CdΔ)

        if any(isnan(gpars)); throw("parameter grad invalid: $gpars"); end
    end

    # tuning curve gradients
    if gxs != nothing
        for (ix, val) in enumerate((Cinv * xs + squeeze(sum(λs .- rs, 3), 3) * κ)[:])
            gxs[ix] = val
        end

        if any(isnan(gxs)); throw("tuning curve grad invalid"); end
    end

    return nll
end

function infer_marginal(rs; Δ0=0.5, κ0=0.5, Δmin=0.1, μmin=0.1)
    # rs: ncond by nnrn by ntrial
    ncond, nnrn, ntrial = size(rs)

    # cost function and associated gradient
    function cost(z)
        # recover the variables
        κ, μ, Δ = z[1], z[2]^2 + μmin, z[3]^2 + Δmin
        # compute the cost
        return approx_marginal_nll!(κ, μ, Δ, rs)
    end
    function cost_g!(z, g::Vector; return_cost=false)
        # recover the variables
        κ, μ, Δ = z[1], z[2]^2 + μmin, z[3]^2 + Δmin
        # grad calculation
        g[:] = 0
        E = approx_marginal_nll!(κ, μ, Δ, rs; gpars=g)
        g = g .* [1, 2 * z[2], 2 * z[3]]

        if any(isinf(g)) || any(isnan(g)); throw("grad invalid"); end

        if return_cost; return E; end
    end
    cost_and_g!(z, g::Vector) = cost_g!(z, g; return_cost=true)

    # reasonable initial guesses for the averaged firing rate
    μ0 = mean(rs[:])
    # initial vector
    z0 = [κ0, sqrt(μ0 - μmin), sqrt(Δ0 - Δmin)]
    # solve
    rst = Optim.optimize(DifferentiableFunction(cost, cost_g!, cost_and_g!), z0, method=:cg)
    #=rst = Optim.optimize(cost, z0, method=:cg, ftol=1e-5)=#
    z = rst.minimum
    # return values
    return z[1], z[2]^2 + μmin, z[3]^2 + Δmin
end

function approx_marginal_nll!(κ, μ, Δ, rs; gpars=nothing)
    ncond, nnrn, ntrial = size(rs)

    rs_μ = squeeze(mean(rs, 3), 3) + 1e-12
    xs_μ = log(rs_μ) / κ - log(μ) / κ + κ / 2
    if any(isnan(xs_μ)) || any(isinf(xs_μ)); throw("invalid xs_μ"); end

    ker = Kernel(Δ)
    C, CdΔ = cov_uniform(ncond, ker), nothing

    nll = 0.0
    if gpars != nothing; CdΔ = GP.deriv_cov_uniform(ncond, ker); end
    for n in 1:nnrn
        Rtilde = diagm(rs_μ[:, n] * ntrial * κ^2)
        Rtildeinv = diagm(1 ./ rs_μ[:, n] / ntrial / κ^2)

        npzwrite("M.npy", Rtildeinv + C)
        M = inv(Rtildeinv + C)
        nll += 0.5 * logdet(eye(ncond) + Rtilde * C) + 0.5 * (xs_μ[:, n]' * M * xs_μ[:, n])[1]
        if isnan(nll) || isinf(nll); throw("invalid nll"); end

        if gpars != nothing
            MM = inv(eye(ncond) + Rtilde * C)

            # κ derivative
            gpars[1] += trace(MM * Rtilde * C) / κ
            gpars[1] += sum((M * xs_μ[:, n]) .* (-xs_μ[:, n] / κ + 1))
            gpars[1] += (xs_μ[:, n]' * M * Rtildeinv * M * xs_μ[:, n] / κ)[1]

            # μ derivative
            gpars[2] += sum(-M * xs_μ[:, n] / κ / μ)

            # Δ dertivative
            gpars[3] += trace(MM * Rtilde * CdΔ) / 2
            gpars[3] -= (xs_μ[:, n]' * M * CdΔ * M * xs_μ[:, n])[1] / 2
        end
    end

    return nll
end

function infer_cov(rs; Δ0=0.5, κ0=0.5, Δmin=0.1, μmin=0.1)
    ncond, nnrn, ntrial = size(rs)
    tmp = reshape(rs, ncond, nnrn * ntrial)
    Cexp_data = log(tmp * tmp' / nnrn / ntrial)

    function cost(pars)
        let κ = pars[1], μ = pars[2]^2 + μmin, Δ = pars[3]^2 + Δmin, ker = Kernel(Δ)
            Cexp = GP.cov_uniform(ncond, ker) * κ^2 + 2log(μ)
            return sum((Cexp - Cexp_data).^2)
        end
    end

    # reasonable initial guesses for the averaged firing rate
    μ0 = mean(rs[:])
    # initial vector
    z0 = [κ0, sqrt(μ0 - μmin), sqrt(Δ0 - Δmin)]
    # solve
    rst = Optim.optimize(cost, z0, method=:nelder_mead)
    z = rst.minimum
    # return values
    return z[1], z[2]^2 + μmin, z[3]^2 + Δmin
end

function infer_loo(rs; Δ0=0.5, κ0=0.5, Δmin=0.1, μmin=0.1)
    # rs: ncond by nnrn by ntrial
    ncond, nnrn, ntrial = size(rs)

    # cost function and associated gradient
    function cost(z)
        # recover the variables
        κ, μ, Δ, xs = z[1], z[2]^2 + μmin, z[3]^2 + Δmin, reshape(z[4:end], ncond, nnrn)
        # compute the cost
        return loo_helper!(κ, μ, Δ, xs, rs)
    end
    function cost_g!(z, g::Vector; return_cost=false)
        # recover the variables
        κ, μ, Δ, xs = z[1], z[2]^2 + μmin, z[3]^2 + Δmin, reshape(z[4:end], ncond, nnrn)
        # calc grad
        g[:] = 0
        E = loo_helper!(κ, μ, Δ, xs, rs; gpars=view(g, 1:3), gxs=view(g, 4:length(g)))
        g[1:3] = g[1:3] .* [1, 2 * z[2], 2 * z[3]]
        # cap gradients
        if any(isinf(g))
            g[isinf(g)] = 1e9 * sign(g[isinf(g)])
        end

        if return_cost; return E; end
    end
    cost_and_g!(z, g::Vector) = cost_g!(z, g; return_cost=true)

    # reasonable initial guesses for the averaged firing rate
    μ0 = mean(rs[:])
	# initialize tuning curve, ncond by nnrn
    xs0 = log(squeeze(mean(rs, 3), 3) / μ0) / κ0 + κ0 / 2
    # initial vector
    z0 = vcat([κ0, sqrt(μ0 - μmin), sqrt(Δ0 - Δmin)], xs0[:])
    # solve
    rst = Optim.cg(DifferentiableFunction(cost, cost_g!, cost_and_g!), z0, grtol=1e-4, ftol=1e-6, iterations=2000, show_trace=false, extended_trace=false)
    z = rst.minimum
    # return values
    return z[1], z[2]^2 + μmin, z[3]^2 + Δmin, reshape(z[4:end], ncond, nnrn)
end

function loo_helper!(κ, μ, Δ, xs, rs; gpars=nothing, gxs=nothing)
    # dimensions
    ncond, nnrn, ntrial = size(rs)

    # convenient functions
    λ2x(λ) = log(λ / μ) / κ + κ / 2
    x2λ(x) = μ * exp(κ * (x - κ / 2))

    # covariance and associated functions
    ker = Kernel(Δ)
    C = cov_uniform(ncond, ker)
    Cinv = inv(C + eye(ncond) * 1e-12)
    CdΔ = GP.deriv_cov_uniform(ncond, ker)
    if any(isinf(CdΔ)) || any(isnan(CdΔ)); throw("Covariance derivative invalid"); end

    # leave one out covariances
    # doing it once due to rotational symmetry of the problem with evenly spaced conditions
    Cnc, Pnc = C[2:end, 2:end], C[1, 2:end]
    Cncinv = inv(Cnc + eye(ncond - 1) * 1e-12)
    CncdΔ, PncdΔ = CdΔ[2:end, 2:end], CdΔ[1, 2:end]

    # conditional variance
    σc = min(1, sqrt(max(1e-12, 1 - (Pnc * Cncinv * Pnc')[1])))
    if isnan(σc) || isinf(σc); throw("invalid conditional variance: $σc"); end
    dσcdΔ = 1 / 2 / σc * (-2 * PncdΔ * Cncinv * Pnc' + Pnc * Cncinv * CncdΔ * Cncinv * Pnc')[1]

    # downshift operator
    R = diagm(ones(ncond - 2), -1); R[1, end] = 1

    # initial values
    cost = 0.0

    mask = ones(Bool, ncond)
    for c in 1:ncond
        # set the skip mask
        mask[c] = false
        mask_ix = find(mask)

        for n in 1:nnrn
            # conditional mean
            μc = (Pnc * Cncinv * xs[mask, n])[1]
            λc = x2λ(μc) + 1e-12
            if isnan(μc) || isinf(μc); throw("invalid conditional mean: $μc"); end
            if isinf(λc) || λc > 1e5; λc = 1e5; end
            # function value
            cost += sum(λc .- rs[c, n, :] * log(λc))
            cost += log(σc) + 0.5 * (xs[c, n] - μc)^2 / σc^2
            if isnan(cost) || isinf(cost); throw("Cost value invalid: $cost at $n"); end

            if gpars != nothing
                # κ
                gpars[1] += sum(λc .- rs[c, n, :]) * (μc - κ)
                # μ
                gpars[2] += sum(λc .- rs[c, n, :]) / μ
                # Δ
                dμcdΔ = (PncdΔ * Cncinv * xs[mask, n] - Pnc * Cncinv * CncdΔ * Cncinv * xs[mask, n])[1]
                gpars[3] += sum(λc .- rs[c, n, :])  * κ * dμcdΔ + (1 / σc - (xs[c, n] - μc)^2 / σc^3) * dσcdΔ - (xs[c, n] - μc) / σc^2 * dμcdΔ

                if any(isnan(gpars)); throw("parameter grad invalid: $gpars"); end
            end

            if gxs != nothing
                gμcgx = Cncinv * Pnc'

                for (i, o) in enumerate(mask_ix + (n - 1) * ncond)
                    gxs[o] += sum(λc .- rs[c, n, :]) * κ * gμcgx[i] - (xs[c, n] - μc) / σc^2 * gμcgx[i]
                end
                gxs[c + (n - 1) * ncond] += (xs[c, n] - μc) / σc^2

                if any(isnan(gxs)); throw("xs grad invalid"); end
            end
        end
        # reset the mask
        mask[c] = true

        # shift the covariances
        Cnc, Pnc = R * Cnc * R', Pnc * R'
        Cncinv = R * Cncinv * R'
        CncdΔ, PncdΔ = R * CncdΔ * R', PncdΔ * R'
    end

    return cost
end

end
