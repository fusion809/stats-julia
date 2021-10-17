#!/usr/bin/env Rscript
# Set up data set
dat <- read.csv("ProjectData.csv")
dat$Group = factor(dat$Group);
attach(dat);

# Linear model
lm.model <- lm(Flight.distance ~ Group);
print("General linear model:")
print("Coefficients:")
print(coef(lm.model))
print(anova(lm.model))
print("----------------------------------------------")

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