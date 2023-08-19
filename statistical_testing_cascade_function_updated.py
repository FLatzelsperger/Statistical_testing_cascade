# Author: Florian Latzelsperger
# Date: 03.2023

def testingCcascade(sample1, sample2, alpha, paired) -> list:
    '''
    Performs statistical tests on two samples (list) in a cascade-like fashion, determining if there is a significant difference.
    Final H0 (simplified): Samples are statistically the same

        Parameters:
            sample1 (list): First sample
            sample2 (list): Second sample
            alpha (float): Significance level to be used in all internal tests
            paired (bool): Specifies if the two samples are paired, or unpaired samples; False = unpaired, True = paired
        
        Returns:
            p-value, final test (list): Mentions the final test that was used in the cascade (index 1) and its corresponding p-value (index 0)

        Caution:
            It is recommended to avoid applying this statistical testing cascade on samples with small sizes below 30.

        Note:
            Based on "Statistik für Biowissenschaftler; A. B. Kesel et al.; 1999; Birkhäuser"

    '''

    ### Load packages
    import scipy.stats as stats

    ### Unpaired samples
    if not paired:
        # 1) Shapiro-Wilk test: Are samples normally distributed? H0: Sample is normally distributed.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html#scipy.stats.shapiro
        p1 = stats.shapiro(sample1).pvalue
        p2 = stats.shapiro(sample2).pvalue

        if (p1 > alpha) and (p2> alpha):
            normality = True
        else:
            normality = False

        # 2) Bartlett's test: Are samples showing the same variance (homoscedasticity)? H0: Samples are from populations with equal variances
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html#scipy.stats.bartlett
        if normality:
            p = stats.bartlett(sample1, sample2).pvalue

            if p < alpha:
                homoscedasticity = False
            else:
                homoscedasticity = True
        
        # 3) Finally test if samples are significantly different (significant difference in sample means). H0: Samples are drawn from populations with the same population means
        # 3.a) t test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
        if normality and homoscedasticity:
            sigTest = 't-test'
            p = stats.ttest_ind(sample1, sample2, equal_var = True).pvalue
        
        # 3.b) Welch test
        # Info: Doing Welch-test here is based on "Statistik für Biowissenschaftler; A. B. Kesel et al.; 1999; Birkhäuser"
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
        if normality and (not homoscedasticity):
            sigTest = 'Welch-test'
            p = stats.ttest_ind(sample1, sample2, equal_var = False).pvalue
        
        # 3.c) U test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
        if not normality:
            sigTest = 'U-test'
            p = stats.mannwhitneyu(sample1, sample2).pvalue
        
        return [p, sigTest]

    ### Paired samples
    if paired:
        if len(sample1) != len(sample2):
            raise Exception('Unable to perform paired tests, unequal sample sizes.')

        # Generate pair-differences
        diff = sample1 - sample2

        # 1) Shapiro-Wilk test: Are pair-differences normally distributed: H0: Pair differences are normally distributed
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html#scipy.stats.shapiro
        p = stats.shapiro(diff).pvalue

        if p < alpha:
            normality = False
        else:
            normality = True
        
        # 2) Finally test if samples paired differences are significantly different from 0
        # 2.a) t test for pair-differences. H0: mean of pair-differences is equal to 0
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html#scipy.stats.ttest_1samp
        if normality:
            sigTest = 'Paired t-test'
            p = stats.ttest_1samp(diff, 0).pvalue

        # 2.b) Wilcoxon test for paired differences. H0: paired samples come from the same distribution
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#scipy.stats.wilcoxon
        if not normality:
            sigTest = 'Wilcoxon test'
            p = stats.wilcoxon(diff).pvalue
        
        return [p, sigTest]

testingCcascade(sample1, sample2, alpha, paired)

### Testing
import numpy as np
import scipy.stats as stats

# Generate test data
rng = np.random.default_rng(seed = 123)

sampleNorm1 = stats.norm.rvs(size = 50, loc = 10, scale = 2, random_state = rng) # Data normally distributed around 10, standard deviation 2, length 50
sampleNorm2 = stats.norm.rvs(size = 50, loc = 10.5, scale = 2, random_state = rng) # Data normally distributed around 10, standard deviation 2, length 50
sampleNorm3 = stats.norm.rvs(size = 50, loc = 10, scale = 5, random_state = rng) # Data normally distributed around 10, standard deviation 5, length 50
sampleNorm4 = stats.norm.rvs(size = 50, loc = 12, scale = 2, random_state = rng) # Data normally distributed around 12, standard deviation 2, length 50
sampleNorm5 = stats.norm.rvs(size = 70, loc = 10, scale = 2, random_state = rng) # Data normally distributed around 20, standard deviation 2, length 70
sampleRndm1 = stats.uniform.rvs(size = 100, loc = 10, random_state = rng) # Data non-normally distributed around 10, length 100

# Test case 1: Unpaired, both normal, similar mean, same variance, same length
testingCcascade(sampleNorm1, sampleNorm2, alpha = 0.05, paired = False)

# Test case 2: Unpaired, both normal, similar mean, different variance, same length
testingCcascade(sampleNorm1, sampleNorm3, alpha = 0.05, paired = False)

# Test case 3: Unpaired, both normal, different mean, same variance, same length
testingCcascade(sampleNorm1, sampleNorm4, alpha = 0.05, paired = False)

# Test case 4: Unpaired, both normal, same mean, same variance, different length
testingCcascade(sampleNorm1, sampleNorm5, alpha = 0.05, paired = False)

# Test case 5: Unpaired, only one sample normal
testingCcascade(sampleNorm1, sampleRndm1, alpha = 0.05, paired = False)

# Test case 6: Paired, providing different sample lenghts (should lead to error)
testingCcascade(sampleNorm1, sampleRndm1, alpha = 0.05, paired = True)

# Test case 6: Paired, providing different sample lenghts (should lead to error)
testingCcascade(sampleNorm1, sampleNorm4, alpha = 0.05, paired = True)