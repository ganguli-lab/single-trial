module LTI

export genDyn, genDynRot, genObsSamp, dlyap, runLTI, xCov, xCovTheory, hankel, hoKalman

function genDyn(N, K, dFast, dSlowLow, dSlowDiff)
  Ds = rand(K) * dSlowDiff + dSlowLow;
  Ds = [Ds; dFast * ones(N - K)];
  U, _ = qr(randn(N, N));
  return real(U * diagm(Ds) * U'), U, Ds
end

function genDynRot(N, K, dFast, dSlowLow, dSlowDiff, omegaScale)
  Khalf = div(K, 2)
  Ds = exp(1im * (rand(Khalf) * 2 - 1)) .* (rand(Khalf) * dSlowDiff + dSlowLow);
  Ds = [Ds; conj(Ds); dFast * ones(N - K)];
  U, _ = qr(randn(N, N));
  U = hcat(U[:, 1:Khalf] + U[:, Khalf + 1:K] * 1im,
           U[:, 1:Khalf] - U[:, Khalf + 1:K] * 1im,
           U[:, K + 1:end] * sqrt(2)) / sqrt(2);
  return real(U * diagm(Ds) * U'), U, Ds
end

# Generate a linear observation matrix that randomly samples M out of N states
function genObsSamp(N, M)
  return eye(N)[randperm(N), :][1:M, :]
end

# solve the discrete Lyapunov equation using an infinite series
# Not checking convergence, not efficient
function dlyap(A, Q)
  X = Q
  while true
    Xnew = Q + A * X * A'
    if norm(Xnew - X) < sqrt(eps(Float64))
      return X
    else
      X = Xnew
    end
  end
end

function runLTI(A, C, Q, R, T::Int64; P=nothing)
  M, N = size(C)

  q = real(sqrtm(Q))
  r = real(sqrtm(R))

  if P == nothing
    p = real(sqrtm(dlyap(A, Q)))
  else
    p = real(sqrtm(P))
  end

  X = zeros(N, T)
  X[:, 1] = p * randn(N)
  for t = 2:T
    X[:, t] = A * X[:, t - 1] + q * randn(N)
  end

  Y = C * X + r * randn(M, T);
  return X, Y
end

function xCov(Y, offset::Int64)
  assert(offset >= 0);
  _, T = size(Y);
  return  Y[:, offset + 1:end] * Y[:, 1:end - offset]' / (T - offset);
end

function xCovTheory(A, C, Q, offset::Int64; P=nothing)
  assert(offset >= 0);
  if P == nothing
    P = dlyap(A, Q);
  end
  return C * A^offset * P * C';
end

function hankel(xCovFun, dly::Int64)
  M, _ = size(xCovFun(1))
  H = zeros(M * dly, M * dly)

  # First block column
  for blkRow = 1:dly
    rowStart = (blkRow - 1) * M + 1
    rowStop = blkRow * M
    H[rowStart:rowStop, 1:M] = xCovFun(blkRow)
  end

  # remaining block columns
  for blkCol = 2:dly
    colStart = (blkCol - 1) * M + 1
    colStop = blkCol * M
    # shifting up the previous blk column
    H[1:end - M, colStart:colStop] = H[M + 1:end, colStart - M:colStop - M]
    # compute only the last block
    H[end - M + 1:end, colStart:colStop] = xCovFun(dly + blkCol - 1)
  end
  return H
end

function hoKalman(Y, dly::Int64, dim::Int64)
  M, _ = size(Y)
  # hankel matrix
  H = hankel(x -> xCov(Y, x), dly)
  # left singular vectors
  d, v = eigs(H * H', nev=dim)
  # observability matrix
  O = v * diagm(d.^0.25)
  # solve the recursive linear regression
  x = O[1:end - M, :]; y = O[M + 1:end, :]
  return (x' * y) * inv(x' * x), H
end
end
