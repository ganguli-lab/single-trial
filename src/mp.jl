# A Marchenko-Pastur type uncorrelated noise model

immutable MPModel <: NoiseModel
  p::Integer
  n::Integer
  c::Number
  sigma::Number
end

## Constructors

function MPModel(p::Integer, n::Integer, sigma::Number)
  c = p / n
  if 0 >= c || 0 >= sigma
    error("c or sigma out of bound")
  else
    return MPModel(p, n, c, sigma)
  end
end

function MPModel(cp::Number, sigma::Number)
  if 0 >= cp || 0 >= sigma
    error("c or phi out of bound")
  end

  if cp < 1
    p, n = 1000::Integer, int(p / cp)
  else
    n, p = 1000::Integer, int(n * cp)
  end
  c = p / n

  if abs(c - cp) > sqrt(eps(Float64)); warn(string("actual c set to ", c)); end

  return MPModel(p, n, c, sigma)
end

## Sampling

function rand(model::MPModel)
  return randn(model.p, model.n) * model.sigma
end

## Eigenvalue and singular value spectrum
##   Wikipedia Marchenko–Pastur distribution

ev_lb(m::MPModel) = m.sigma^2 * (1 - sqrt(m.c))^2 * min(m.n, m.p)

ev_ub(m::MPModel) = m.sigma^2 * (1 + sqrt(m.c))^2 * min(m.n, m.p)

function ev_spec(m::MPModel)
  l, u = ev_lb(m), ev_ub(m)
  x -> sqrt((u - x) * (x - l)) / m.c / x / 2 / pi / m.sigma^2
end

sv_lb(m::MPModel) = sqrt(ev_lb(m))

sv_ub(m::MPModel) = sqrt(ev_ub(m))

function sv_spec(m::MPModel)
  es = ev_spec(m)
  x -> es(x^2) * 2 * x
end

## Low-rank perturbations. see:
##   Benaych-Georges, F. & Nadakuditi, R., 2012
##   Gavish, M. & Donoho D. L., 2014

ev_sigthresh(m::MPModel) = m.sigma^2 * sqrt(m.p * m.n)

sv_sigthresh(m::MPModel) = sqrt(ev_sigthresh(m))

function sv_xfer(m::MPModel, s::Number)
  let sigma = m.sigma, c = m.c, n = m.n
    if s < sv_sigthresh(m); return sv_ub(m); end
    sp = s / sigma / sqrt(n)
    sqrt((sp + 1 / sp) * (sp + c / sp)) * sigma * sqrt(n)
  end
end

ev_xfer(m::MPModel, ev::Number) = sv_xfer(m, sqrt(ev))^2

function svec_overlap(m::MPModel, s::Number)
  let sigma = m.sigma, c = m.c, n = m.n
    if s <= sv_sigthresh(m)
      l, r = 0, 0
    else
      sp = s / sigma / sqrt(n)
      l = sqrt((sp^4 - c) / (sp^4 + c * sp^2))
      r = sqrt((sp^4 - c) / (sp^4 + sp^2))
    end
    (l, r)
  end
end
