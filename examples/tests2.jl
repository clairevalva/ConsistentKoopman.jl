ym = X[13,:]
ym_1 = X[12,:]
yn = X[2743,:]
yn_1= X[2742,:]

testbw = sepBandwidth(ym, yn, ym_1, yn_1)

n_m_normdiff = (yn - ym) ./ norm(yn - ym)
m_n_normdiff = (ym - yn) ./ norm(yn - ym)

vm_normed = (ym - ym_1) ./ norm(ym - ym_1)
vn_normed = (yn - yn_1) ./ norm(yn - yn_1)

costheta_m = vm_normed' * n_m_normdiff
costheta_n = vn_normed' * m_n_normdiff

ζ_constant = 0.995
anstest = sqrt((1 - ζ_constant*costheta_m) * (1 - ζ_constant*costheta_n))

testker = varbandwidth_kernel(ym, yn, ym_1, yn_1, γ = useϵ)

kerans = exp(-1*(norm(ym - yn)*anstest / useϵ)^2)
