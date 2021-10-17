#!/usr/bin/env julia
using StatsFuns;

"""
    oneWayANOVA(m::Int64, n::Int64, nvec::Matrix{Int64}, 
    yarr::Matrix{BigFloat})

Perform one-way ANOVA test.

Parameters
----------
`m::Int64`: number of treatment groups.

`n::Int64`: number of observations.

`nvec::Matrix{Int64}`: m x 1 matrix of sample sizes.

`yarr::Matrix{BigFloat}`: m x ni matrix of values of the dependent variable.

Returns
-------
Nothing.
"""
function oneWayANOVA(m, n, nvec, yarr)
    CM = 1/n * sum(yarr)^2;
    totalSS = sum(yarr.^2) - CM;
    SST = sum(sum(yarr, dims=2).^2 .* nvec.^(-1)) - CM
    SSE = totalSS - SST;
    MST = SST/(m-1);
    MSE = SSE/(n-m);
    F = MST/MSE;
    pval = 1-fdistcdf(m-1, n-m, Float64(F));

    # Print results
    println("One-way ANOVA test:")
    println("MSE            = ", Float64(MSE))
    println("MST            = ", Float64(MST))
    println("F              = ", Float64(F))
    println("Numerator df   = ", m-1)
    println("Denominator df = ", n-m)
    println("P-value        = ", pval)
    println("--------------------------------------------------")
end