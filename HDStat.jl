module HDStat

export HDModel, randSpec, randSpecDensity, logSpecSupport, linSpecSupport

abstract HDModel

global const epsilon = 1e-9

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

function randSpecDensity(model::HDModel, nTrial::Integer, nBin::Integer)
  D = randSpec(model, nTrial)
  bins = linspace(minimum(D), maximum(D), nBin)
  _, counts = hist(D, bins)
  counts /= length(D) * mean(diff(bins))

  return (bins[2:end] + bins[1:end - 1]) * 0.5, counts
end

function logSpecSupport(model::HDModel, n::Integer)
  lower, upper = lb(model), ub(model)

  return logspace(log10(lower + epsilon), log10(upper - epsilon), n)
end

function linSpecSupport(model::HDModel, n::Integer)
  lower, upper = lb(model), ub(model)

  return linspace(lower + epsilon, upper - epsilon, n)
end

end
