type DynModel
    phis::Number
    phin::Number
    sigmas::Number
    Ws::Array{Float64, 2}
    Us::Array{Float64, 2}
    Un::Array{Float64, 2}
    S::Array{Float64, 2}
    N::Integer
    K::Integer
    M::Integer
    T::Integer
end

type NLDynModel
    base::DynModel
    scale::Number
end

function DynModel(N, K, M, T, phis, phin, sigmas)
    Ws = genDynRot(K, K, 0, phis, 0, pi / 2)[1]
    U = qr(randn(N, N))[1]
    Us = U[:, 1:K]
    Un = U[:, K + 1:end]
    S = eye(N)[randperm(N)[1:M], :]

    DynModel(phis, phin, sigmas, Ws, Us, Un, S, N, K, M, T)
end

function sample(m::DynModel)
    C = eye(m.K);
    Q = eye(m.K) * m.sigmas^2;
    R = eye(m.K) * 0;
    p = eye(m.K) / (1 - m.phis^2) * m.sigmas^2
    Xs = runLTI(m.Ws, C, Q, R, m.T; P=p)[1]
    m.S * (m.Us * Xs + m.Un * rand(ARMPModel(m.N - m.K, m.T, m.phin, 1)))
end

function sample(m::NLDynModel)
    C = eye(m.base.N)
    Q = m.base.Us * eye(m.base.K) * m.base.Us' * m.base.sigmas^2 + m.base.Un * m.base.Un'
    R = eye(m.base.N) * 0;
    p = m.base.Us * (eye(m.base.K) ./ (1 - m.base.phis^2) * m.base.sigmas^2) * m.base.Us'
    p += m.base.Un * (eye(m.base.N - m.base.K) / (1 - m.base.phin^2)) * m.base.Un'
    W = m.base.Us * m.base.Ws * m.base.Us' + m.base.Un * diagm(ones(m.base.N - m.base.K) * m.base.phin) * m.base.Un'
    Xs = runTanh(W, C, Q, R, m.scale, m.base.T; P=p)[1]
    m.base.S * Xs
end

function fullSample(m::DynModel)
    C = eye(m.K)
    Q = eye(m.K) * m.sigmas^2
    R = eye(m.K) * 0
    p = eye(m.K) / sqrt(1 - m.phis^2) * m.sigmas
    Xs = runLTI(m.Ws, C, Q, R, m.T; P=p)[1]
    Xf = m.Us * Xs + m.Un * rand(ARMPModel(m.N - m.K, m.T, m.phin, 1))
    X = m.S * Xf
    return {:Xs => Xs, :Xf => Xf, :X => X}
end
