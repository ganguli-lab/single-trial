immutable StaticModel <: DataModel
  noise::NoiseModel
  sig::SigModel
end

# sampling

rand(m::StaticModel) = rand(m.sig) + rand(m.noise)
