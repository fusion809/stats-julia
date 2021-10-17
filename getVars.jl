#!/usr/bin/env julia
using LinearAlgebra, Statistics;

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