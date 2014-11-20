immutable ObsModel <: Model
  m::Integer # num of observed neurons
  n::Integer # total num of neurons
  sig::SigModel # signal model
  noise::NoiseModel # noise model
  S::Array{Float64, 2} # random sampling matrix
end

# constructor

function ObsModel(m, sig::SigModel, noise::NoiseModel)
  if sig.n != noise.n || sig.p != noise.p
    error("signal and noise dimensionality mismatch")
  end

  S = eye(sig.n)[randperm(sig.n)[1:m], :]
  ObsModel(m, sig.n, sig, noise, S)
end

# sampling

rand(o::ObsModel) = o.S * (rand(o.sig) + rand(o.noise))

function randfull(o::ObsModel)
  rst = merge(randfull(o.sig), randfull(o.noise))
  rst[:Rf] = rst[:Z] + rst[:X]
  rst[:Rs] = o.S * rst[:Rf]
  rst
end
