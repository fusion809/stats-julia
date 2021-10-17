#!/usr/bin/env julia
# Written in October 2021 by Brenton Horne
using StatsFuns, CSV, DataFrames, Statistics, SpecialFunctions, LinearAlgebra
using RCall

"""
    getVars(group::Vector{Int64}, y::Vector{Float64})

Compute various required variables from the independent and dependent 
variables. 

Parameters
----------
`group::Vector{Int64}`: Contains values of the grouping (independent) variable.

`y::Vector{Float64}`: Contains values of the dependent (response) variable.

Returns
-------
`m::Int64`: number of groups.

`n::Int64`: total number of observations.

`ni::Int64`: maximum sample size.

`alphavec::Matrix{BigFloat}`: m x 1 matrix of initial guess values for the 
maximum likelihood estimator (MLE) of ``\\alpha_i``.

`nvec::Matrix{Int64}`: m x 1 matrix of the sample sizes of each group.

`yarr::Matrix{BigFloat}`: m x ni matrix of the observations, with the m rows
corresponding to different values of the grouping variable.

`ybarvec::Matrix{BigFloat}`: m x 1 matrix of the mean of each group.
"""
function getVars(group, y)
    # Number of groups should equal the number of unique values of the grouping
    # variable
    m = Int(length(unique(group)))
    
    # nvec's elements should equal the number of observations for each value of
    # the grouping variable.
    nvec = zeros((m, 1))
    for i=1:m
        nvec[i, 1] = length(y[isequal.(group, i)])
    end
    nvec = Int.(nvec)

    # Simpler to calculate variables
    n = length(y)
    ni = maximum(nvec)

    # Put observations from y into rows based on the corresponding value of the
    # grouping variable
    yarr = BigFloat.(zeros((m, ni)))
    for i=sort(unique(group))
        for j=1:nvec[i, 1]
            yarr[i,j] = y[isequal.(group, i)][j]
        end
    end

    # Means
    ybar = mean(y)
    ybarvec = mean(yarr, dims=2)
    ybarvec = reshape(ybarvec, m, 1)

    # Let's use a simple initial guess for the MLE of alpha_i, we know it's 
    # greater than 0
    alphavec = 10 * BigFloat.(ones((m, 1)))

    return m, n, ni, alphavec, nvec, yarr, ybar, ybarvec
end

"""
    funjacUnr(alphavec::Matrix{BigFloat}, nvec::Matrix{Int64}, 
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
function funjacUnr(m, alphavec, nvec, yarr, ybarvec)
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
    funjacNull(alpha::BigFloat, n::Int64, yarr::Matrix{BigFloat}, 
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
function funjacNull(alpha, n, yarr, ybar)
    J = n * (1/alpha - polygamma(1, Float64(alpha)))
    F = -n * (polygamma(0, Float64(alpha)) + log(ybar/alpha)) + sum(log.(yarr))
    return J, F
end

function funjacNormal(mu, var, nvec, yarr, ybarvec)
    muHat = sum(nvec .* ybarvec .* var.^(-1))/(sum(nvec.*var.^(-1)));
    f = mu - muHat;
    g = var .- nvec.^(-1) .* sum((yarr.-mu).^2, dims=2)
    F = append!([f], g);
    m = length(var);
    J0 = nvec.*(var.^2 .*sum(nvec.*var.^(-1))).^(-1).*(ybarvec.-muHat);
    J0 = append!([BigFloat(1.0)], J0);
    J = Matrix(1.0I, m+1, m+1);
    J[1, :] = J0;
    J[2:m+1, 1] = 2*(ybarvec.-mu)

    return J, F;
end

function newtonsNormal(m, muNull, varNull, nvec, yarr, ybarvec, itMax, tol)
    J, F = funjacNormal(muNull, varNull, nvec, yarr, ybarvec);
    eps = -J\F;
    diff = sqrt(sum(eps.^2)/(m+1));

    iteration = 0;
    param = zeros((m+1, 1));
    
    while (tol < diff && iteration < itMax)
        muNull += eps[1];
        varNull += eps[2:m+1];
        J, F = funjacNormal(muNull, varNull, nvec, yarr, ybarvec);
        eps = -J\F;
        param[1] = muNull;
        param[2:m+1] = varNull;
        epsRel = eps .* param.^(-1);
        diff = sqrt(sum(epsRel.^2)/(m+1));
        iteration += 1;
    end

    return muNull, varNull;
end

"""
    newtonsUnr(m::Int64, alphavec::Matrix{BigFloat}, nvec::Matrix{Int64}, 
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
function newtonsUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol)
    # Obtain what vars we need for iteration 1
    Jinv, F = funjacUnr(m, alphavec, nvec, yarr, ybarvec)
    eps = -Jinv * F
    epsRel = eps .* alphavec.^(-1)
    diff = sqrt(sum(epsRel.^2)/m)

    # Initialize our counter
    iteration = 0

    # Iterate until we get a satisfactorily accurate estimate for the MLE
    while ((tol < diff) && (iteration < itMax))
        alphavec += eps
        Jinv, F = funjacUnr(m, alphavec, nvec, yarr, ybarvec)
        eps = -Jinv * F
        epsRel = eps .* alphavec.^(-1)
        diff = sqrt(sum((epsRel.^2)/m))
        iteration += 1
    end

    return alphavec
end

"""
    newtonsNull(alpha::BigFloat, n::Int64, yarr::Matrix, ybar::Float64, 
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
function newtonsNull(alpha, n, yarr, ybar, itMax, tol)
    # Obtain what vars we need for iteration 1
    J, F = funjacNull(alpha, n, yarr, ybar)
    eps = -F/J
    
    # Initialize iteration counter
    iteration = 0

    # Iterate until we get a satisfactorily accurate for the MLE
    while ((tol < eps/alpha) && (iteration < itMax))
        alpha += eps
        J, F = funjacNull(alpha, n, yarr, ybar)
        eps = -F/J
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
    alphavec = newtonsUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol)
    betavec = alphavec.^(-1) .* ybarvec

    # Estimate MLEs under null
    alpha = 1
    alpha = newtonsNull(alpha, n, yarr, ybar, itMax, tol)
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

function normalTest(m, ybar, nvec, ybarvec, yarr, itMax, tol)
    muNull, varNull = newtonsNormal(m, ybar, var(yarr, dims=2), nvec, yarr, ybarvec, itMax, tol);
    varUnrest = nvec.^(-1) .* sum((yarr.-ybarvec).^2, dims=2);
    lam = prod((varUnrest.*varNull.^(-1)).^(nvec/2))
    stat = -2*log(lam);
    pval = 1-chisqcdf(m-1, Float64(stat));
    return muNull, varNull, varUnrest, lam, stat, pval;
end

function printNormal(m, muNull, varNull, ybarvec, varUnrest, lam, stat, pval)
    println("For normal model:")
    println("Mean (null)               = ", Float64(muNull))
    println("Variance (null)           = ", Float64.(varNull))
    println("Mean (unrestricted)       = ", Float64.(ybarvec))
    println("Variance (unrestricted)   = ", Float64.(varUnrest))
    println("Lambda                    = ", Float64(lam))
    println("Test statistic            = ", Float64(stat))
    println("Degrees of freedom        = ", m-1)
    println("P-value                   = ", pval)
    println("----------------------------------------------")
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
    println("Lambda             = ", lam)
    println("Test statistic     = ", Float64(stat))
    println("Degrees of freedom = ", 2*m-2)
    println("P-value            = ", pval)
    println("----------------------------------------------")
end

"""
    printExp(m::Int64, n::Int64, nvec::Matrix{Int64}, ybar::BigFloat, 
    ybarvec::Matrix{BigFloat})

Print the results of the exponential likelihood-ratio test after performing it.

Parameters
----------
`m::Int64`: number of groups.

`n::Int64`: total number of observations.

`nvec::Matrix{Int64}`: m x 1 matrix of sample sizes.

`ybar::BigFloat`: overall mean of all observations.

`ybar::Matrix{BigFloat}`: m x 1 matrix of the mean of each sample (treatment 
group).

Returns
-------
Nothing.
"""
function printExp(m, n, nvec, ybar, ybarvec)
    # Exponential distribution test
    statExp = 2*n*log(ybar) - 2*sum(nvec .* log.(ybarvec))
    pvalExp = 1-chisqcdf(m-1, Float64(statExp))
    println("For exponential model:")
    println("Test statistic     = ", Float64(statExp))
    println("Degrees of freedom = ", m - 1)
    println("P-value            = ", pvalExp)
    println("----------------------------------------------")
end

"""
    main()

Calls other functions to extract the required data, perform hypothesis tests 
and print the results.
"""
function main()
    # Get problem data and parameters
    csv_reader = CSV.File("ProjectData.csv")
    dataF = DataFrame(csv_reader)

    # Get y and group
    y = BigFloat.(dataF[:, 6])
    group = dataF[:, 1]
    m, n, ni, alphavec, nvec, yarr, ybar, ybarvec = getVars(group, y)
    itMax = 1e3;
    tol = 1e-13;

    # Use R to perform ANOVA and GLM analysis
    R"source('anovaGLM.R')"

    # Perform the normal test
    muNull, varNull, varUnrest, lamNorm, statNorm, pvalNorm = normalTest(m, 
    ybar, nvec, ybarvec, yarr, itMax, tol);

    # Perform the gamma test
    alpha, beta, alphavec, betavec, lamGam, statGam, pvalGam = gammaTest(m, n,
    ni, alphavec, nvec, group, yarr, ybar, ybarvec, itMax, tol);

    # Print gamma and exp test MLEs and results
    println("Likelihood-ratio tests:")
    printGamma(m, alpha, beta, alphavec, betavec, lamGam, statGam, pvalGam);
    printExp(m, n, nvec, ybar, ybarvec);
    printNormal(m, muNull, varNull, ybarvec, varUnrest, lamNorm, statNorm, 
    pvalNorm)
end

if isinteractive()
    main()
end