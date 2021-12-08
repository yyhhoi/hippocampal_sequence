import os
import numpy as np
from scipy.special import comb
from scipy.stats import chi2_contingency, kruskal, chisquare, ttest_1samp, ranksums, f as fdist, fisher_exact
import scikit_posthocs as posthoc_tests

from pycircstat import watson_williams, cdiff, resultant_vector_length
from pycircstat.descriptive import std as circstd, mean as circmean


def p2str(pval):
    if pval >= 0.0001:
        return '{:.4f}'.format(pval)
    else:
        return '{:.1e}'.format(pval)


def fdr_bh_correction(pvals):
    m = len(pvals)
    ranks = np.argsort(pvals) + 1
    qvals = pvals * (m / ranks)
    qvals[qvals > 1] = 1
    return qvals.squeeze().tolist()


def stat_record(fn, overwrite, *args):
    rootdir = 'writting/stats'
    os.makedirs(rootdir, exist_ok=True)
    pth = os.path.join(rootdir, fn)

    if overwrite:
        if os.path.isfile(pth):
            os.remove(pth)

    with open(pth, 'a') as fh:
        print_list = []
        for arg in args:

            if isinstance(arg, str):
                new_arg = arg
            elif (arg is None) or (np.isnan(arg)):
                new_arg = 'None'
            else:
                new_arg = str(np.around(arg, 4))

            print_list.append(new_arg)

        txt = ', '.join(print_list)
        fh.write(txt + '\n')
    return None


def wwtable2text(wwtable):
    df_col = wwtable.loc['Columns', 'df']
    df_Residual = wwtable.loc['Residual', 'df']
    Fstat = wwtable.loc['Columns', 'F']
    pval = wwtable.loc['Columns', 'p-value']
    text = 'F(%d, %d)=%0.2f, p%s' % (df_col, df_Residual, Fstat, p2str(pval))
    return text


def fisherexact_directmethod(arr):
    """

    Parameters
    ----------
    arr : ndarray
        2 x 2 numpy array, as [[a, b], [c, d]]

    Returns
    -------
    p : float
        p-value or Fisher's exact test.
    """
    a, b = arr[0, 0], arr[0, 1]
    c, d = arr[1, 0], arr[1, 1]
    n = a + b + c + d
    up = comb(a + b, a) * comb(c + d, c)
    down = comb(n, a + c)
    p = up / down
    if np.isnan(p):
        _, p = fisher_exact(arr)
    return p


def category2waytest(data_arr):
    n_samples = np.sum(data_arr)
    chi_stat, chi_p, chi_dof, _ = chi2_contingency(data_arr)
    pfisher = fisherexact_directmethod(data_arr)
    chi_text = '\chi^2(%d, N=%d)=%0.2f, p%s' % (chi_dof, n_samples - 2, chi_stat, p2str(chi_p))
    fisher_text = "Fisher's exact test, p%s" % (p2str(pfisher))
    return (chi_text, chi_p), (fisher_text, pfisher)


def ranksums2text(arr1, arr2, tag1, tag2):
    N1, N2 = arr1.shape[0], arr2.shape[0]
    Ustat, p_rs = ranksums(arr1, arr2)
    rs_text = 'U(N_{%s}=%d, N_{%s}=%d)=%0.2f, p%s' % (tag1, N1, tag2, N2, Ustat, p2str(p_rs))
    return Ustat, p_rs, rs_text


def my_kruskal_2samp(x1, x2, tag1, tag2):
    n1 = len(x1)
    n2 = len(x2)
    sample_sizes = (n1, n2)
    mdn1, mdn2 = np.median(x1), np.median(x2)
    lqr1, hqr1 = np.quantile(x1, 0.25), np.quantile(x1, 0.75)
    lqr2, hqr2 = np.quantile(x2, 0.25), np.quantile(x2, 0.75)
    kruskal_results = kruskal(x1, x2)
    Hstat, kruskal_p = kruskal_results
    descrips = ((mdn1, lqr1, hqr1), (mdn2, lqr2, hqr2))
    txt = r'Kruskal-Wallis, %s $(n=%d)$ vs %s $(n=%d)$, $H_{(1)}=%0.2f, p=%s$' % \
          (tag1, n1, tag2, n2, Hstat, p2str(kruskal_p))
    return kruskal_p, sample_sizes, descrips, txt


def my_kruskal_3samp(x1, x2, x3, tag1, tag2, tag3):
    n1 = len(x1)
    n2 = len(x2)
    n3 = len(x3)
    sample_sizes = (n1, n2, n3)
    mdn1, mdn2, mdn3 = np.median(x1), np.median(x2), np.median(x3)
    lqr1, hqr1 = np.quantile(x1, 0.25), np.quantile(x1, 0.75)
    lqr2, hqr2 = np.quantile(x2, 0.25), np.quantile(x2, 0.75)
    lqr3, hqr3 = np.quantile(x3, 0.25), np.quantile(x3, 0.75)
    kruskal_results = kruskal(x1, x2, x3)
    Hstat, kruskal_p = kruskal_results

    dunntable = posthoc_tests.posthoc_dunn([x1, x2, x3], p_adjust='fdr_bh').to_numpy()
    p12 = dunntable[0, 1]
    p23 = dunntable[1, 2]
    p13 = dunntable[0, 2]
    dunn_pvals = (p12, p23, p13)

    ktxt = r'Kruskal-Wallis, %s $(n=%d)$ vs %s $(n=%d)$ vs %s $(n=%d)$, $H_{(2)}=%0.2f, p=%s$' % \
           (tag1, n1, tag2, n2, tag3, n3, Hstat, p2str(kruskal_p))

    dunntxt12 = r'%s vs %s, $p=%s$' % (tag1, tag2, p2str(p12))
    dunntxt23 = r'%s vs %s, $p=%s$' % (tag2, tag3, p2str(p23))
    dunntxt13 = r'%s vs %s, $p=%s$' % (tag1, tag3, p2str(p13))
    dunntxt = r'\textit{post hoc} Dunn’s test with Benjamini-Hochberg correction: %s, %s, %s' % (
    dunntxt12, dunntxt23, dunntxt13)

    txt = ktxt + '; ' + dunntxt

    descrips = ((mdn1, lqr1, hqr1), (mdn2, lqr2, hqr2), (mdn3, lqr3, hqr3))

    return kruskal_p, dunn_pvals, sample_sizes, descrips, txt


def my_ww_2samp(x1, x2, tag1, tag2):
    n1, n2 = len(x1), len(x2)
    sample_sizes = (n1, n2)

    cmean1, cmean2 = circmean(x1), circmean(x2)
    sem1, sem2 = circstd(x1) / np.sqrt(n1), circstd(x2) / np.sqrt(n2)
    descrips = ((cmean1, sem1), (cmean2, sem2))

    ww_results = watson_williams(x1, x2)
    ww_pval, wwtable = ww_results
    df_col = wwtable.loc['Columns', 'df']
    df_Residual = wwtable.loc['Residual', 'df']
    Fstat = wwtable.loc['Columns', 'F']
    pval = wwtable.loc['Columns', 'p-value']
    txt = r'Watson-Williams, %s $(n=%d)$ vs %s $(n=%d)$, $F_{(%d, %d)}=%0.2f, p=%s$' % (
    tag1, n1, tag2, n2, df_col, df_Residual, Fstat, p2str(pval))
    return ww_pval, sample_sizes, descrips, txt


def my_fisher_2way(data_arr):
    n = np.sum(data_arr)
    fisher_pval = fisherexact_directmethod(data_arr)
    txt = r"Fisher\vtick{s} exact, $n=%d, p=%s$" % (n, p2str(fisher_pval))
    return fisher_pval, n, txt


def my_chi_2way(data_arr):
    n = np.sum(data_arr)
    try:
        chi_stat, chi_pval, chi_dof, _ = chi2_contingency(data_arr)
        txt = r'Chi-square for independence, $\chi^{2}_{(%d, %d)}=%0.2f, p=%s$' % (
        chi_dof, n, chi_stat, p2str(chi_pval))
        return chi_pval, n, txt
    except ValueError:
        return 1, n, 'Unavailable as some entry has 0 value.'


def my_chi_1way(freqs):
    n = np.sum(freqs)
    chi_stat, chi_pval = chisquare(freqs)
    txt = r'One-way Chi-square test, $\chi^{2}_{(1, %d)}=%0.2f, p=%s$' % (n, chi_stat, p2str(chi_pval))
    return chi_pval, n, txt


def my_ttest_1samp(x, exp_x=0):
    n = len(x)
    xmean = np.mean(x)
    ttest_stat, p_1d1samp = ttest_1samp(x, exp_x)

    txt = r"Student's t-test, mean=%0.4f, $t_{(%d)}=%0.2f, p=%s$" % (xmean, n - 1, ttest_stat, p2str(p_1d1samp))
    return p_1d1samp, n, txt


def angular_dispersion_test(alpha1, alpha2):
    mean1 = circmean(alpha1)
    mean2 = circmean(alpha2)
    adv1 = np.abs(cdiff(alpha1, mean1))
    adv2 = np.abs(cdiff(alpha2, mean2))
    z, pval = ranksums(adv1, adv2)
    return z, pval, (adv1, adv2)


def circ_ktest(alpha1, alpha2):
    alpha1 = np.asarray(alpha1)
    alpha2 = np.asarray(alpha2)

    n1 = alpha1.shape[0]
    n2 = alpha2.shape[0]

    R1 = resultant_vector_length(alpha1)
    R2 = resultant_vector_length(alpha2)

    f_stat = ((n2 - 1) * (n1 - R1)) / ((n1 - 1) * (n2 - R2))

    if f_stat > 1:
        pval = 2 * (1 - fdist.cdf(f_stat, n1, n2))
    else:
        f_stat = 1 / f_stat
        pval = 2 * (1 - fdist.cdf(f_stat, n2, n1))

    return f_stat, pval
