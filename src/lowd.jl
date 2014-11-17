immutable LowDModel <: SigModel
  k::Integer # dimensionality
  p::Integer # embedding dimensions
  n::Integer # num of data points
  sigma::Number # variance
  U::Array{Number, 2} # embedding operator, R^k -> R^p
end

# constructors

function LowDModel(k::Integer, p::Integer, n::Integer, sigma::Number)
  U = qr(randn(p, k))[1]
  LowDModel(k, p, n, sigma, U)
end

# sampling

rand(m::LowDModel) = m.U * randn(m.k, m.p) * m.sigma
