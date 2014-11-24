# A Marchenko-Pastur type uncorrelated noise model

immutable MPModel <: NoiseModel
  n::Integer # number of neurons
  p::Integer # number of trials
  c::Number # shape = n / p
  sigma::Number
end

## Constructors

function MPModel(n::Integer, p::Integer, sigma::Number)
  c = n / p
  if 0 >= c || 0 >= sigma
    error("c or sigma out of bound")
  else
    MPModel(n, p, c, sigma)
  end
end

function MPModel(cp::Number, sigma::Number)
  if 0 >= cp || 0 >= sigma
    error("c or sigma out of bound")
  end

  if cp < 1
    n, p = 1000::Integer, int(1000 / cp)
  else
    p, n = 1000::Integer, int(1000 * cp)
  end
  c = n / p

  if abs(c - cp) > epsilon; warn(string("actual c set to ", c)); end

  MPModel(n, p, c, sigma)
end

## Sampling

rand(m::MPModel) = randn(m.n, m.p) * m.sigma

randfull(m::MPModel) = {:Z => rand(m)}

## Eigenvalue and singular value spectrum
##   Wikipedia Marchenko–Pastur distribution

ev_lb(m::MPModel) = m.sigma^2 * abs(sqrt(m.n) - sqrt(m.p))^2

ev_ub(m::MPModel) = m.sigma^2 * (sqrt(m.n) + sqrt(m.p))^2

# TODO: might need better handling at the lower bound of support, when p == n
function ev_spec(m::MPModel)
  l, u = ev_lb(m), ev_ub(m)
  x -> sqrt((u - x) * (x - l)) / x / 2 / pi / m.sigma^2 / min(m.n, m.p)
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

ev_infloor(m::MPModel) = m.sigma^2 * sqrt(m.n * m.p)

sv_infloor(m::MPModel) = sqrt(ev_infloor(m))

sv_outfloor(m::MPModel) = sv_xfer(m, sv_infloor(m))

ev_outfloor(m::MPModel) = sv_outfloor(m)^2

function sv_xfer(m::MPModel, s::Number)
  let sigma = m.sigma, c = m.c, p = m.p
    if s < sv_infloor(m); return sv_ub(m); end
    sp = s / sigma / sqrt(p)
    sqrt((sp + 1 / sp) * (sp + c / sp)) * sigma * sqrt(p)
  end
end

ev_xfer(m::MPModel, ev::Number) = sv_xfer(m, sqrt(ev))^2

function svec_overlap(m::MPModel, s::Number)
  let sigma = m.sigma, c = m.c, p = m.p
    if s <= sv_infloor(m)
      l, r = 0, 0
    else
      sp = s / sigma / sqrt(p)
      l = sqrt((sp^4 - c) / (sp^4 + c * sp^2))
      r = sqrt((sp^4 - c) / (sp^4 + sp^2))
    end
    (l, r)
  end
end
