module LTI

export genDyn, genDynRot, genObsSamp, dlyap, runLTI, xCov, hankel, hoKalman

function genDyn(N, K, dFast, dSlowLow, dSlowDiff)
  Ds = rand(K) * dSlowDiff + dSlowLow;
  Ds = [Ds; dFast * ones(N - K)];
  U, _ = qr(randn(N, N));
  return real(U * diagm(Ds) * U'), U, Ds
end

function genDynRot(N, K, dFast, dSlowLow, dSlowDiff, omegaScale)
  Ds = exp(1im * (rand(div(K, 2)) * 2 - 1)) .* (rand(div(K, 2)) * dSlowDiff + dSlowLow);
  Ds = [Ds; conj(Ds); dFast * ones(N - K)];
  U, _ = qr(randn(N, N));
  U = hcat(U[:, 1:K / 2] + U[:, K / 2 + 1:K] * 1im,
           U[:, 1:K / 2] - U[:, K / 2 + 1:K] * 1im,
           U[:, K + 1:end] * sqrt(2)) / sqrt(2);
  return real(U' * diagm(Ds) * U), U, Ds
end

function genObsSamp(N, M)
  return eye(N)[randperm(N), :][1:M, :]
end

function dlyap(A, Q)
  X = Q
  while true
    Xnew = Q + A * X * A'
    if norm(Xnew - X) < 1e-14
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
  _, T = size(Y)
  return Y[:, offset + 1:end] * Y[:, 1:end - offset]' / (T - offset)
end

function hankel(Y, dly::Int64)
  M, T = size(Y)
  H = zeros(M * delay, M * delay)

  # First block column
  for blkRow = 1:dly
    rowStart = (blkRow - 1) * M + 1
    rowStop = blkRow * M
    H[rowStart:rowStop, 1:M] = xCov(Y, blkRow)
  end

  # remaining block columns
  for blkCol = 2:dly
    colStart = (blkCol - 1) * M + 1
    colStop = blkCol * M
    # shifting up the previous blk column
    H[1:end - M, colStart:colStop] = H[M + 1:end, colStart - M:colStop - M]
    # compute only the last block
    H[end - M + 1:end, colStart:colStop] = xCov(Y, dly + blkCol - 1)
  end
  return H
end

function hoKalman(Y, dly::Int64, dim::Int64)
  # hankel matrix
  H = hankel(Y, dly)
  # left singular vectors
  d, v = eigs(H * H', nev=dim)
  # observability matrix
  O = v * diagm(d.^0.25)
  # solve the recursive linear regression
  x = O[1:end - M, :]; y = O[M + 1:end, :]
  return y \ x
end
end
