using StatsFuns, LinearAlgebra

"""
    expTest(m::Int64, n::Int64, nvec::Matrix{Int64}, ybar::BigFloat, 
    ybarvec::Matrix{BigFloat})

Perform the exponential likelihood-ratio test.

Parameters
----------
`m::Int64`: number of groups.

`n::Int64`: total number of observations.

`nvec::Matrix{Int64}`: m x 1 matrix of sample sizes.

`ybar::BigFloat`: overall mean of all observations.

`ybarvec::Matrix{BigFloat}`: m x 1 matrix of the mean of each sample (treatment
group).

Returns
-------
`theta::BigFloat`: maximum likelihood estimator (MLE) of `\\theta` under the 
null hypothesis.

`thetavec::Matrix{BigFloat}`: unrestricted MLE of `\\theta_i`.

`lam::BigFloat`: likelihood ratio for exponential test.

`stat::BigFloat`: test statistic for exponential test.

`pval::Float64`: p-value for exponential test.
"""
function expTest(m, n, nvec, ybar, ybarvec)
    # MLEs
    thetavec = ybarvec;
    theta = ybar;
    # Likelihood-ratio
    lam = ybar^(-n) * prod(ybarvec.^(nvec));
    # Test statistic and p-value
    stat = -2*log(lam);
    pval = 1-chisqcdf(m-1, Float64(stat))

    return theta, thetavec, lam, stat, pval
end

"""
    printExp(m::Int64, n::Int64, nvec::Matrix{Int64}, ybar::BigFloat, 
    ybarvec::Matrix{BigFloat})

Print the results of the exponential likelihood-ratio test after performing it.

Parameters
----------
`m::Int64`: number of groups.

`theta::BigFloat`: maximum likelihood estimator (MLE) of `\\theta` under the 
null hypothesis.

`thetavec::Matrix{BigFloat}`: unrestricted MLE of `\\theta_i`.

`lam::BigFloat`: likelihood ratio for exponential test.

`stat::BigFloat`: test statistic for exponential test.

`pval::Float64`: p-value for exponential test.

Returns
-------
Nothing.
"""
function printExp(m, theta, thetavec, lam, stat, pval)
    # Exponential distribution test
    println("For exponential model:")
    println("theta              = ", Float64(theta))
    println("theta_i            = ", Float64.(thetavec))
    println("lambda             = ", Float64(lam))
    println("Test statistic     = ", Float64(stat))
    println("Degrees of freedom = ", m - 1)
    println("P-value            = ", pval)
    println("--------------------------------------------------")
end