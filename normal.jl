#!/usr/bin/env julia
using StatsFuns, Statistics, LinearAlgebra

"""
    funjacNormal(mu::BigFloat, var::Matrix{BigFloat}, nvec::Matrix{Int64}, 
    yarr::Matrix{BigFloat}, ybarvec::Matrix{BigFloat})

Compute the Jacobian and function values required to apply Newton's method to 
compute the maximum likelihood estimators (MLEs) under the null hypothesis for 
normally-distributed populations with non-constant variances. See
https://github.com/fusion809/LRT-normal-nonconst-var/blob/master/doc.pdf for
the mathematical details.

Parameters
----------
`m::Int64`: number of treatment groups.

`mu::BigFloat`: current best estimate of the MLE of `\\mu` under the null.

`var::Matrix{BigFloat}`: m x 1, current best estimate of the MLE of 
`\\sigma_i^2` under the null.

`nvec::Matrix{Int64}`: m x 1, contains sample sizes for each treatment group.

`yarr:Matrix{BigFloat}`: m x ni, contains dependent variable values for each 
observation categorized by treatment group (different row, different group).

`ybarvec::Matrix{BigFloat}`: m x 1, contains the group sample means.

Returns
-------
`J::Matrix{BigFloat}`: (m+1) x (m+1), Jacobian of problem.

`F::Matrix{BigFloat}`: (m+1) x 1 of function values we're trying to set to 0.
"""
function funjacNormal(m, mu, var, nvec, yarr, ybarvec)
    # Create function value matrix
    muHat = sum(nvec .* ybarvec .* var.^(-1))/(sum(nvec.*var.^(-1)));
    f = mu - muHat;
    g = var .- nvec.^(-1) .* sum((yarr.-mu).^2, dims=2)
    F = append!([f], g);

    # Create Jacobian
    J0 = nvec.*(var.^2 .*sum(nvec.*var.^(-1))).^(-1).*(ybarvec.-muHat);
    J0 = append!([BigFloat(1.0)], J0);
    J = Matrix(1.0I, m+1, m+1);
    J[1, :] = J0;
    J[2:m+1, 1] = 2*(ybarvec.-mu)

    return J, F;
end

"""
    newtonsNormal(m::Int64, muNull::BigFloat, varNull::Matrix{BigFloat}, 
    nvec::Matrix{Int64}, yarr::Matrix{BigFloat}, ybarvec::Matrix{BigFloat}, 
    itMax::Int64, tol::Float64)

Apply Newton's method to approximate the maximum likelihood estimators (MLEs)
for `\\mu` and `\\sigma_i^2` under the null hypothesis of equality of means
of the normally-distributed populations our samples are presumed to come from. 

Parameters
----------
`m::Int64`: number of treatment groups. 

`muNull::BigFloat`: initial estimate of the MLE of `\\mu` under the null.

`varNull::Matrix{BigFloat}`: m x 1, initial estimate of the MLE of 
`\\sigma_i^2` under the null.

`nvec::Matrix{Int64}`: m x 1, contains sample sizes for each treatment group.

`yarr:Matrix{BigFloat}`: m x ni, contains dependent variable values for each 
observation categorized by treatment group (different row, different group).

`ybarvec::Matrix{BigFloat}`: m x 1, contains the group sample means.

`itMax::Int64`: maximum number of iterations of Newton's method that can be 
used in estimating our MLEs under the null.

`tol::Float64`: the relative error tolerance we are going to use for Newton's.

Returns
-------
`muNull::BigFloat`: our MLE of `\\mu` under the null hypothesis.

`varNull::Matrix{BigFloat}`: our MLE of `\\sigma_i^2` under the null 
hypothesis.
"""
function newtonsNormal(m, muNull, varNull, nvec, yarr, ybarvec, itMax, tol)
    # First iteration of Newton's
    J, F = funjacNormal(m, muNull, varNull, nvec, yarr, ybarvec);
    eps = -J\F;
    param = zeros((m+1, 1));
    muNull += eps[1];
    varNull += eps[2:m+1];
    param[1] = muNull;
    param[2:m+1] = varNull;
    epsRel = eps .* param.^(-1);
    diff = sqrt(sum(epsRel.^2)/(m+1));
    
    # Initialize iteration counter
    iteration = 1;

    # Iteratively apply Newton's method until our MLE estimate is satisfactory    
    while (tol < diff && iteration < itMax)
        J, F = funjacNormal(m, muNull, varNull, nvec, yarr, ybarvec);
        eps = -J\F;
        muNull += eps[1];
        varNull += eps[2:m+1];
        param[1] = muNull;
        param[2:m+1] = varNull;
        epsRel = eps .* param.^(-1);
        diff = sqrt(sum(epsRel.^2)/(m+1));
        iteration += 1;
    end

    return muNull, varNull;
end

"""
    normalTest(m::Int64, ybar::Matrix{BigFloat}, nvec::Matrix{Int64}, 
    ybarvec::Matrix{BigFloat}, yarr::Matrix{BigFloat}, itMax::Int64, 
    tol::Float64)

Perform the normal non-constant variance likelihood-ratio test. 

Parameters
----------
`m::Int64`: number of treatment groups. 

`ybar::BigFloat`: mean of all observations.

`nvec::Matrix{Int64}`: m x 1, contains sample sizes for each treatment group.

`ybarvec::Matrix{BigFloat}`: m x 1, contains the group sample means.

`yarr:Matrix{BigFloat}`: m x ni, contains dependent variable values for each 
observation categorized by treatment group (different row, different group).

`itMax::Int64`: maximum number of iterations of Newton's method that can be 
used in estimating our MLEs under the null.

`tol::Float64`: the relative error tolerance we are going to use for Newton's.

Returns
-------
`muNull::BigFloat`: our MLE of `\\mu` under the null hypothesis.

`varNull::Matrix{BigFloat}`: our MLE of `\\sigma_i^2` under the null 
hypothesis.

`varUnrest::Matrix{BigFloat}`: our unrestricted MLE of `\\sigma_i^2`.

`lam::BigFloat`: our likelihood ratio.

`stat::BigFloat`: our test statistic, -2ln(lam), which under the null 
hypothesis should be chi-squared distributed with df=m-1.

`pval::Float64`: our p-value.
"""
function normalTest(m, ybar, nvec, ybarvec, yarr, itMax, tol)
    # Compute our MLEs
    muNull, varNull = newtonsNormal(m, ybar, var(yarr, dims=2), nvec, yarr, ybarvec, itMax, tol);
    varUnrest = nvec.^(-1) .* sum((yarr.-ybarvec).^2, dims=2);

    # Likelihood ratio
    lam = prod((varUnrest.*varNull.^(-1)).^(nvec/2))

    # Test statistic and p-value
    stat = -2*log(lam);
    pval = 1-chisqcdf(m-1, Float64(stat));
    return muNull, varNull, varUnrest, lam, stat, pval;
end

"""
    printNormal(m::Int64, muNull::BigFloat, varNull::Matrix{BigFloat}, 
    ybarvec::Matrix{BigFloat}, varUnrest::Matrix{BigFloat}, lam::BigFloat, 
    stat::BigFloat, pval::Float64)

Print the maximum likelihood estimators (MLEs) of the parameters, along with 
likelihood ratio, test statistic and p-value for the likelihood-ratio test 
that assumes normally-distributed populations for our treatment groups and 
does not assume equal variances across treatment groups.

Parameters
----------
`m::Int64`: number of treatment groups. 

`muNull::BigFloat`: our MLE of `\\mu` under the null hypothesis.

`varNull::Matrix{BigFloat}`: our MLE of `\\sigma_i^2` under the null 
hypothesis.

`ybarvec::Matrix{BigFloat}`: m x 1, contains the group sample means.

`varUnrest::Matrix{BigFloat}`: our unrestricted MLE of `\\sigma_i^2`.

`lam::BigFloat`: our likelihood ratio.

`stat::BigFloat`: our test statistic, -2ln(lam), which under the null 
hypothesis should be chi-squared distributed with df=m-1.

`pval::Float64`: our p-value.

Returns
-------
Nothing. 
"""
function printNormal(m, muNull, varNull, ybarvec, varUnrest, lam, stat, pval)
    println("For normal model:")
    println("mu (null)                 = ", Float64(muNull))
    println("sigma_i^2 (null)          = ", Float64.(varNull))
    println("mu_i (unrestricted)       = ", Float64.(ybarvec))
    println("sigma_i^2 (unrestricted)  = ", Float64.(varUnrest))
    println("lambda                    = ", Float64(lam))
    println("Test statistic            = ", Float64(stat))
    println("Degrees of freedom        = ", m-1)
    println("P-value                   = ", pval)
    println("--------------------------------------------------")
end