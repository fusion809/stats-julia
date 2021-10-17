#!/usr/bin/env julia
# Written in October 2021 by Brenton Horne
using CSV, DataFrames, RCall

# Load required functions
include("getVars.jl")
include("gamma.jl")
include("normal.jl")
include("exp.jl")

# Get problem data and parameters
csv_reader = CSV.File("ProjectData.csv")
dataF = DataFrame(csv_reader)

# Get y and group
y = BigFloat.(dataF[:, 6])
group = dataF[:, 1]
m, n, ni, alphavec, nvec, yarr, ybar, ybarvec = getVars(group, y)
itMax = 1e3
tol = 1e-13

# Use R to perform ANOVA and GLM analysis
R"source('anovaGLM.R')"

# Perform the normal test
muNull, varNull, varUnrest, lamNorm, statNorm, pvalNorm = normalTest(m, ybar, 
nvec, ybarvec, yarr, itMax, tol)

# Perform the gamma test
alpha, beta, alphavec, betavec, lamGam, statGam, pvalGam = gammaTest(m, n,ni, 
alphavec, nvec, group, yarr, ybar, ybarvec, itMax, tol)

# Perform the exponential test
theta, thetavec, lamExp, statExp, pvalExp = expTest(m, n, nvec, ybar, 
ybarvec)

# Print gamma and exp test MLEs and results
println("Likelihood-ratio tests:")
printGamma(m, alpha, beta, alphavec, betavec, lamGam, statGam, pvalGam)
printExp(m, theta, thetavec, lamExp, statExp, pvalExp)
printNormal(m, muNull, varNull, ybarvec, varUnrest, lamNorm, statNorm, 
pvalNorm)