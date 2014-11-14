export MPModel, rand, ub, lb, spec, vecOverlap, eigvalXfer, pertThresh

# A Marchenko-Pastur type uncorrelated noise model
immutable MPModel <: NoiseModel
  p::Integer
  n::Integer
  c::Float64
  sigma::Float64
end

## constructors

function MPModel(p::Integer, n::Integer, sigma::Number)
  c = (p / n)
  if !(0 < c && 0 < sigma)
    error("c or sigma out of bound")
  else
    return MPModel(p, n, c, sigma)
  end
end

function MPModel(cp::Number, sigma::Number)
  if !(0 < cp && 0 < sigma)
    error("c or phi out of bound")
  end
  if cp < 1
    p = 1000::Int64
    n = int(p / cp)
  else
    n = 1000::Int64
    p = int(n * cp)
  end
  c = p / n
  if abs(c - cp) > 1e-6
    warn(string("actual c set to ", c))
  end
  return MPModel(p, n, c, sigma)
end

## Sampling

function rand(model::MPModel)
  return randn(model.p, model.n) * model.sigma
end

## Eigenvalue spectrum (not singular values!!)

function lb(model::MPModel)
  return model.sigma^2 * (1 - sqrt(model.c))^2
end

function ub(model::MPModel)
  return model.sigma^2 * (1 + sqrt(model.c))^2
end

function spec(model::MPModel)
  low, up = lb(model), ub(model)
  return x -> 0.5 / pi / model.sigma^2 * sqrt(up - x) * sqrt(x - low) / model.c / x
end

## Low-rank perturbations

function pertThresh(model::MPModel)
  return model.sigma^2 * sqrt(model.p / model.n)
end

function vecOverlap(model::MPModel)
  let sigma = model.sigma, c = model.c
    return x -> x / sigma^2 < sqrt(c) ? (0, 0) : (sqrt((x^2 - c * sigma^2) / (x^2 + c * sigma^2 * x)), sqrt((x^2 - c * sigma^2) / (x^2 + x)))
  end
end

function eigvalXfer(model::MPModel)
  let sigma = model.sigma, c = model.c
    return x -> x / sigma^2 < sqrt(c) ? ub(model) : (sqrt(x) + 1 / sqrt(x)) * (sqrt(x) + c * sigma^2 / sqrt(x))
  end
end
