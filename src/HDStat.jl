module HDStat

# models
export NoiseModel, SigModel, ObsModel
export MPModel, ARMPModel
export LowDModel
# sampling
export rand, randfull, sv_rand, ev_rand
# spectrum
export ev_lb, ev_ub, sv_lb, sv_ub, ev_spec, sv_spec
export ev_linsupport, ev_logsupport, sv_linsupport, sv_logsupport
# low-rank perturbations
export ev_infloor, ev_outfloor, ev_xfer, sv_infloor, sv_outfloor, sv_xfer, svec_overlap
# randSpec, randSpecDensity, logSpecSupport, linSpecSupport, pertThresh

abstract Model

abstract NoiseModel <: Model

abstract SigModel <: Model

global const epsilon = sqrt(eps(Float64))

include("mp.jl") # Marchenko-Pastur
include("armp.jl") # Marchenko-Pastur-like for autoregressive model
include("lowd.jl") # Low-dimensional signal
include("obs.jl") # Observation data model

# sample singular and eiven value spectra given a noise model

sv_rand(m::Model) = [svd(rand(m))[2], zeros(max(0, m.n - m.p))]

ev_rand(m::Model) = sv_rand(m).^2

# return the support of singular or eigen value spectra given a noise model

ev_linsupport(m::NoiseModel, k::Integer) = linspace(ev_lb(m) + epsilon, ev_ub(m) - epsilon, k)

ev_logsupport(m::NoiseModel, k::Integer) = logspace(log10(ev_lb(m) + epsilon), log10(ev_ub(m) - epsilon), k)

sv_linsupport(m::NoiseModel, k::Integer) = linspace(sv_lb(m) + epsilon, sv_ub(m) - epsilon, k)

sv_logsupport(m::NoiseModel, k::Integer) = logspace(log10(sv_lb(m) + epsilon), log10(sv_ub(m) - epsilon), k)

end
