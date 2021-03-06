#!/usr/bin/env Rscript
# Written by Brenton Horne in October 2021
# Set up data set
dat <- read.csv("ProjectData.csv")
dat$Group = factor(dat$Group);
attach(dat);

# Welch's ANOVA
print("Welch's ANOVA:")
print(oneway.test(Flight.distance ~ Group))
print("----------------------------------------------")

# Gamma generalized linear model
glm.model <- glm(Flight.distance ~ Group, family=Gamma(link="inverse"));
print("Gamma (inverse link) generalized linear model:")
print("Coefficients:")
print(coef(glm.model))
print(anova(glm.model, test="Chisq"));
print("----------------------------------------------")