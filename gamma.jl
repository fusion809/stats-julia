using StatsFuns, Statistics, SpecialFunctions, LinearAlgebra

"""
    funjacGamUnr(alphavec::Matrix{BigFloat}, nvec::Matrix{Int64}, 
    yarr::Matrix{BigFloat}, ybarvec::Matrix{BigFloat})

Compute the inverse of the Jacobian and the matrix of function values for 
computing the unrestricted maximum likelihood estiamtor (MLE) of ``\\alpha_i``.

Parameters
----------
`alphavec::Matrix{BigFloat}`: m x 1 matrix of values of our current estimate 
of the MLE of ``\\alpha_i``.

`nvec::Matrix{Int64}`: m x 1 matrix of sample sizes of each group.

`yarr::Matrix{Float64}`: m x ni matrix of observations in rows that correspond
to different values of the grouping variable.

`ybarvec::Matrix{Float64}`: m x 1 matrix of sample means. 

Returns
-------
`Jinv::Diagonal{BigFloat, Vector{BigFloat}}`: inverse of the Jacobian for the
unrestricted MLE problem.

`F::Matrix{BigFloat}`: m x 1 matrix of function values for the unrestricted MLE
problem.
"""
function funjacGamUnr(m, alphavec, nvec, yarr, ybarvec)
    # Jinv is a diagonal matrix composed of elements of 
    # 1/(partial g_k/partial alpha_i)
    # Alphavec must be converted to Float64 for polygamma,
    # for some reason it cannot handle BigFloat.
    Jinv = Diagonal(vec((nvec .* (alphavec.^(-1)-
    polygamma.(1, Float64.(alphavec)))).^(-1)))
    # Defining logsum to make F's definition shorter
    logsum = reshape(sum(log.(yarr), dims=2), (m, 1))
    F = - nvec .* (polygamma.(0, Float64.(alphavec)) + 
    log.(ybarvec .* alphavec.^(-1))) + logsum

    return Jinv, F
end

"""
    funjacGamNull(alpha::BigFloat, n::Int64, yarr::Matrix{BigFloat}, 
    ybar::BigFloat)

Compute the derivative and function value required to compute the maximum 
likelihood estimator (MLE) of ``\\alpha`` under the null hypothesis.

Parameters
----------
`alpha::BigFloat`: current estimate of the MLE of `\\alpha`.

`n::Int64`: total number of observations.

`yarr::Matrix{BigFloat}`: m x ni matrix of observations with different rows
corresponding to different values of the grouping variable.

`ybar::BigFloat`: mean of all observations.

Returns
-------
`J::BigFloat`: function derivative value.

`F::BigFloat`: function value.
"""
function funjacGamNull(alpha, n, yarr, ybar)
    J = n * (1/alpha - polygamma(1, Float64(alpha)))
    F = -n * (polygamma(0, Float64(alpha)) + log(ybar/alpha)) + sum(log.(yarr))
    return J, F
end

"""
    newtonsGamUnr(m::Int64, alphavec::Matrix{BigFloat}, nvec::Matrix{Int64}, 
    yarr::Matrix{BigFloat}, ybarvec::Matrix{BigFloat}, itMax::Int64, 
    tol::Float64)

Estimate the unrestricted maximum likelihood estimator (MLE) of ``\\alpha_i`` 
using Newton's method.

Parameters
----------
`m::Int64`: number of groups.

`alphavec::Matrix{BigFloat}`: m x 1 matrix of values of our initial estimate 
of the MLE of ``\\alpha_i``.

`nvec::Matrix{Int64}`: m x 1 matrix of sample sizes of each group.

`yarr::Matrix{Float64}`: m x ni matrix of observations in rows that correspond
to different values of the grouping variable.

`ybarvec::Matrix{Float64}`: m x 1 matrix of sample means. 

`itMax::Int64`: maximum number of iterations of Newton's method that can be 
used.

`tol::Float64`: relative error tolerance.

Returns
-------
`alphavec::Matrix{BigFloat}`: m x 1 matrix of values of our improved estimate 
of the MLE of ``\\alpha_i``.
"""
function newtonsGamUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol)
    # First iteration of Newton's method
    Jinv, F = funjacGamUnr(m, alphavec, nvec, yarr, ybarvec)
    eps = -Jinv * F
    alphavec += eps
    epsRel = eps .* alphavec.^(-1)
    diff = sqrt(sum(epsRel.^2)/m)

    # Initialize our counter
    iteration = 1

    # Iterate until we get a satisfactorily accurate estimate for the MLE
    while ((tol < diff) && (iteration < itMax))
        Jinv, F = funjacGamUnr(m, alphavec, nvec, yarr, ybarvec)
        eps = -Jinv * F
        alphavec += eps
        epsRel = eps .* alphavec.^(-1)
        diff = sqrt(sum((epsRel.^2)/m))
        iteration += 1
    end

    return alphavec
end

"""
    newtonsGamNull(alpha::BigFloat, n::Int64, yarr::Matrix, ybar::Float64, 
    itMax::Int64, tol::Float64)

Estimate the maximum likelihood estimator (MLE) of alpha under the null using 
Newton's method.

Parameters
----------
`alpha::BigFloat`: initial estimate of the MLE of `\\alpha`.

`n::Int64`: total number of observations.

`yarr::Matrix{BigFloat}`: m x ni matrix of observations with different rows
corresponding to different values of the grouping variable.

`ybar::BigFloat`: mean of all observations.

Returns
-------
`alpha::BigFloat`: improved estimate of the MLE of `\\alpha`.
"""
function newtonsGamNull(alpha, n, yarr, ybar, itMax, tol)
    # First iteration of Newton's
    J, F = funjacGamNull(alpha, n, yarr, ybar)
    eps = -F/J
    alpha += eps
    
    # Initialize iteration counter
    iteration = 1

    # Iterate until we get a satisfactorily accurate for the MLE
    while ((tol < eps/alpha) && (iteration < itMax))
        J, F = funjacGamNull(alpha, n, yarr, ybar)
        eps = -F/J
        alpha += eps
        iteration +=1
    end
    
    return alpha
end

"""
    gammaTest(m::Int64, n::Int64, ni::Int64, alphavec::Matrix{BigFloat}, 
    nvec::Matrix{Int64}, group::Vector{Int64}, yarr::Matrix{BigFloat}, 
    ybar::BigFloat, ybarvec::Matrix{BigFloat})

Perform the gamma likelihood-ratio test and return the maximum likelihood 
estimator (MLEs), likelihood ratio, test statistic and p-value.

Parameters
----------
`m::Int64`: number of groups.

`n::Int64`: total number of observations.

`ni::Int64`: maximum sample size.

`alphavec::Matrix{BigFloat}`: m x 1 matrix of MLE of ``\\alpha_i``.

`nvec::Matrix{Int64}`: m x 1 matrix of sample sizes for each group.

`group::Vector{Int64}`: vector of values of the grouping variable.

`yarr::Matrix{BigFloat}`: m x ni matrix of observed values of dependent 
variable.

`ybar::BigFloat`: overall mean of dependent variable.

`ybarvec::Matrix{BigFloat}`: m x 1 matrix of the mean of each sample 
(treatment group).

Returns
-------
`alpha::BigFloat`: the MLE of ``\\alpha`` under the null.

`beta::BigFloat`: the MLE of ``\\beta`` under the null.

`alphavec::BigFloat`: the unrestricted MLE of ``\\alpha_i``.

`betavec::BigFloat`: the unrestricted MLE of ``\\beta_i``.

`lam::BigFloat`: ``\\lambda``, the likelihood-ratio. 

`stat::BigFloat`: ``-2\\ln(\\lambda)``, our test statistic.

`pval::BigFloat`: p-value of our test.
"""
function gammaTest(m, n, ni, alphavec, nvec, group, yarr, ybar, ybarvec, itMax, tol)
    # Estimate unrestricted MLEs
    alphavec = newtonsGamUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol)
    betavec = alphavec.^(-1) .* ybarvec

    # Estimate MLEs under null
    alpha = 1
    alpha = newtonsGamNull(alpha, n, yarr, ybar, itMax, tol)
    beta = ybar/alpha

    # Likelihood ratio
    lam = (gamma(alpha) * (ybar/alpha)^(alpha))^(-n)
    lam *= prod(prod(yarr.^(alpha*ones(length(alphavec), 1)-alphavec), dims=2))
    lam *= prod((gamma.(alphavec) .* (ybarvec.*alphavec.^(-1)).^(alphavec)).^(nvec))

    # Test statistic, -2 ln(lambda)
    stat = -2*log(lam)

    # Obtain p-value keeping in mind that under the null our test statistic
    # should asymptotically follow a chi-squared distribution with 2m-2 df
    pval = 1-chisqcdf(2*m-2, Float64(stat))

    return alpha, beta, alphavec, betavec, lam, stat, pval
end

"""
    printGamma(m::Int64, alpha::BigFloat, beta::BigFloat, 
    alphavec::Matrix{BigFloat}, betavec::Matrix{BigFloat}, lam::BigFloat, 
    stat::BigFloat, pval::BigFloat)

Print the results of the gamma likelihood-ratio test. 

Parameters
----------
`m::Int64`: the number of groups.

`alpha::BigFloat`: the maximum likelihood estimator (MLE) of ``\\alpha`` 
under the null.

`beta::BigFloat`: the MLE of ``\\beta`` under the null.

`alphavec::Matrix{BigFloat}`: m x 1 matrix of the unrestricted MLE of 
``\\alpha_i``.

`betavec::Matrix{BigFloat}`: m x 1 matrix of the unrestricted MLE of 
``\\beta_i``.

`lam::BigFloat`: ``\\lambda``, the likelihood-ratio.

`stat::BigFloat`: ``-2\\ln{\\lambda}``, the test statistic.

`pval::BigFloat`: p-value of our test.

Returns
-------
Nothing.
"""
function printGamma(m, alpha, beta, alphavec, betavec, lam, stat, pval)
    # Printing important data
    println("For gamma model:")
    println("alpha (null)       = ", Float64(alpha))
    println("beta (null)        = ", Float64(beta))
    println("alpha_i            = ", Float64.(alphavec))
    println("beta_i             = ", Float64.(betavec))
    println("lambda             = ", lam)
    println("Test statistic     = ", Float64(stat))
    println("Degrees of freedom = ", 2*m-2)
    println("P-value            = ", pval)
    println("--------------------------------------------------")
end
