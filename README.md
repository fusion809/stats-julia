# Statistics in Julia
My original plan for this repository was to perform all the one-way statistical tests that I've performed on my STA3300 experimental project data using a single Julia script. Unfortunately, I found the available statistics libraries for Julia left a lot to be desired, as I found both GLM and MixedAnova difficult to use to perform the analysis of deviance I would like to perform on my gamma generalized linear model (very easy to get type errors). When I overcame these errors, I received a p-value that was heavily rounded, hence making it difficult to fairly compare to the p-values I obtained from my likelihood-ratio tests. ANOVA tests had similar issues in Julia. So I decided to write an R script to perform the gamma GLM analysis of deviance and linear model ANOVA tests and call this R script within my Julia script.

As of commit #15, running solver.jl on my machine which has ProjectData.csv in this repository returns:

```
One-way ANOVA test:
MSE            = 847249.0740720462
MST            = 9.367742888877727e6
F              = 11.05665757042909
Numerator df   = 5
Denominator df = 24
P-value        = 1.3137465285417704e-5
--------------------------------------------------
[1] "Welch's ANOVA:"

        One-way analysis of means (not assuming equal variances)

data:  Flight.distance and Group
F = 44.746, num df = 5.0000, denom df = 9.8678, p-value = 1.814e-06

[1] "----------------------------------------------"
[1] "Gamma (inverse link) generalized linear model:"
[1] "Coefficients:"
  (Intercept)        Group2        Group3        Group4        Group5 
 0.0004838710 -0.0002543377 -0.0003178867 -0.0002664482 -0.0001491989 
       Group6 
-0.0002241757 
Analysis of Deviance Table

Model: Gamma, link: inverse

Response: Flight.distance

Terms added sequentially (first to last)


      Df Deviance Resid. Df Resid. Dev Pr(>Chi)    
NULL                     29     4.4395             
Group  5   3.2693        24     1.1702 4.62e-13 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[1] "----------------------------------------------"
Likelihood-ratio tests:
For gamma model:
alpha (null)       = 6.920003525101537
beta (null)        = 575.2887242847139
alpha_i            = [260.2153578949539; 10.037267625499302; 135.9829866885517; 19.92578566433435; 21.075057963702424; 43.43537680346101]
beta_i             = [7.942139477797275; 434.0490688521165; 44.30456201455098; 230.8231861374744; 141.7789694886204; 88.65277453653455]
lambda             = 3.330888278399484311755809283606544074030899921482052806807963055535231452810561e-988
Test statistic     = 4547.501665718853
Degrees of freedom = 10
P-value            = 0.0
--------------------------------------------------
For exponential model:
theta              = 3981.000000001386
theta_i            = [2066.66666666666; 4356.6666666674655; 6024.6666666668; 4599.333333334066; 2988.000000006672; 3850.666666666652]
lambda             = 0.19502067240847595
Test statistic     = 3.269299427372287
Degrees of freedom = 5
P-value            = 0.6585451040846756
--------------------------------------------------
For normal model:
mu (null)                 = 3780.603400653011
sigma_i^2 (null)          = [2.9541702392189084e6; 2.3486133308947557e6; 5.292920386316882e6; 1.6786449248917922e6; 1.0331250395946143e6; 368017.3056889539]
mu_i (unrestricted)       = [2066.66666666666; 4356.6666666674655; 6024.6666666668; 4599.333333334066; 2988.000000006672; 3850.666666666652]
sigma_i^2 (unrestricted)  = [16591.111111109014; 2.0167644444435148e6; 257100.44444440812; 1.0083262222238667e6; 404904.8888784736; 363108.4444444556]
lambda                    = 2.184006034590533e-11
Test statistic            = 49.09455040392952
Degrees of freedom        = 5
P-value                   = 2.122662379200335e-9
--------------------------------------------------
```