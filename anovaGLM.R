# Set up data set
dat <- read.csv("ProjectData.csv")
dat$Group = factor(dat$Group);
attach(dat);

# Linear model
lm.model <- lm(Flight.distance ~ Group);
print("General linear model:")
print(coef(lm.model))
print(anova(lm.model))

# Welch's ANOVA
print("Welch's ANOVA:")
print(oneway.test(Flight.distance ~ Group))

# Gamma generalized linear model
glm.model <- glm(Flight.distance ~ Group, family=Gamma(link="inverse"));
print("Gamma (inverse link) generalized linear model:")
print(coef(glm.model))
print(anova(glm.model, test="Chisq"));