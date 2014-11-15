module HDStat

# models
export NoiseModel, MPModel
# sampling
export rand
# spectrum
export ev_lb, ev_ub, sv_lb, sv_ub, ev_spec, sv_spec
export ev_linsupport, ev_logsupport, sv_linsupport, sv_logsupport
# low-rank perturbations
export ev_sigthresh, ev_xfer, sv_sigthresh, sv_xfer, svec_overlap
# randSpec, randSpecDensity, logSpecSupport, linSpecSupport, pertThresh

abstract NoiseModel

global const epsilon = sqrt(eps(Float64))

include("mp.jl")
# include("armp.jl")

using Cubature

function dtransform(mu, l, u, c)
  function D(z)
    tmp = pquadrature(t -> sqrt(z) / (z - t) * mu(t), l + 1e-6, u - 1e-6; reltol=1e-3, abstol=1e-3)[1]
    return tmp * (c * tmp + (1 - c) / sqrt(z))
  end
  return D
end

function pertThresh(mu, l, u, c)
  1.0 / dtransform(mu, l, u, c)(u + 1e-4)
end

function pertThresh(model::NoiseModel)
  pertThresh(spec(model), lb(model), ub(model), model.c)
end

# sample the spectrum of a model nTrial times
function randSpec(model::NoiseModel, nTrial::Integer)
  let p = model.p, n = model.n, c = model.c
    if c < 1
      D = Array(Float64, p * nTrial)
    else
      D = Array(Float64, n * nTrial)
    end
    for kTrial in 1:nTrial
      x = rand(model)
      if c < 1
        d, _ = eig(x * x' / n)
        D[(kTrial - 1) * p + 1:kTrial * p] = d
      else
        d, _ = eig(x' * x / n)
        D[(kTrial - 1) * n + 1:kTrial * n] = d
      end
    end
    return D
  end
end

function randSpecDensity(model::NoiseModel, nTrial::Integer, nBin::Integer)
  D = randSpec(model, nTrial)
  bins = linspace(minimum(D), maximum(D), nBin)
  _, counts = hist(D, bins)
  counts /= length(D) * mean(diff(bins))

  return (bins[2:end] + bins[1:end - 1]) * 0.5, counts
end

ev_linsupport(m::NoiseModel, k::Integer) = linspace(ev_lb(m) + epsilon, ev_ub(m) - epsilon, k)

ev_logsupport(m::NoiseModel, k::Integer) = logspace(log10(ev_lb(m) + epsilon), log10(ev_ub(m) - epsilon), k)

sv_linsupport(m::NoiseModel, k::Integer) = linspace(sv_lb(m) + epsilon, sv_ub(m) - epsilon, k)

sv_logsupport(m::NoiseModel, k::Integer) = logspace(log10(sv_lb(m) + epsilon), log10(sv_ub(m) - epsilon), k)

end
