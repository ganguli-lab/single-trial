immutable LowDModel <: SigModel
  k::Integer # dimensionality
  n::Integer # embedding dimensions
  p::Integer # num of data points
  sigma::Number # variance
  U::Array{Number, 2} # embedding operator, R^k -> R^n
end

# constructors

function LowDModel(k::Integer, n::Integer, p::Integer, sigma::Number)
  U = qr(randn(n, k))[1]
  LowDModel(k, n, p, sigma, U)
end

# sampling

rand(m::LowDModel) = m.U * randn(m.k, m.p) * m.sigma

function randfull(m::LowDModel)
  X0 = randn(m.k, m.p)
  {:X0 => X0, :X => m.U * X0}
end
