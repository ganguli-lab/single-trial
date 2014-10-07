module HDStat

abstract HDModel

include("mp.jl")
include("armp.jl")

# sample the spectrum of a model nTrial times
function randSpec(model::HDModel, nTrial::Integer)
  let p = model.p, n = model.n, c = model.c
    D = Array(Float64, p * nTrial)
    for kTrial in 1:nTrial
      x = rand(model)
      d, _ = eig(x * x' / n)
      D[(kTrial - 1) * p + 1:kTrial * p] = d
    end
    return D
  end
end

function logSupport(model::HDModel, n::Integer)
  lower, upper = lb(model), ub(model)

  return logspace(log10(lower + 1e-6), log10(upper + 1e-6), n)
end

function logSupport(model::HDModel, n::Integer)
  lower, upper = lb(model), ub(model)

  return linspace(lower + 1e-6, upper + 1e-6, n)
end

end
