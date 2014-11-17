immutable ObsModel
  m::Integer # num of observed neurons
  n::Integer # total num of neurons
  sig::SigModel # signal model
  noise::NoiseModel # noise model
  S::Array{Number, 2} # random sampling matrix
end

# constructor

function ObsModel(m, sig::SigModel, noise::NoiseModel)
  if sig.n != noise.n || sig.p != noise.p
    error("signal and noise dimensionality mismatch")
  end

  S = eye(sig.n)[randperm(n)[1:m], :]
  ObsModel(m, sig.n, sig, noise, S)
end

# sampling

rand(o::ObsModel) = S * (rand(o.sig) + rand(o.noise))
