"""Build the paper figure set from the saved results of both corpora."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

plt.rcParams.update({'figure.dpi': 600, 'font.size': 8, 'font.family': 'DejaVu Sans',
                     'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .7, 'legend.frameon': False, 'savefig.bbox': 'tight'})


def panel_label(ax, text, fontsize=8, dy=-0.30):
    """Panel label BELOW the axes, as MDPI requires, instead of a title above it."""
    ax.text(0.5, dy, text, transform=ax.transAxes, ha='center', va='top', fontsize=fontsize)


RES = '/home/claude/paper/results'
OUT = '/mnt/user-data/outputs/figures'
os.makedirs(OUT, exist_ok=True)
NOM = 0.95
COL = {'conservative': '#2E6FAD', 'neutral': '#4C9F70', 'aggressive': '#D08C34', 'extreme': '#A33B3B'}
DS = {'4g': f'{RES}/4g', '5g': f'{RES}/5g'}


def save(fig, name):
    # Springer requires >=300 dpi for halftones and >=600 dpi for combination art, plus a vector
    # master; we emit both so the submission can use either.
    fig.savefig(f'{OUT}/{name}.png', dpi=600)
    fig.savefig(f'{OUT}/{name}.pdf')                      # vector master, fonts embedded
    try:
        fig.savefig(f'{OUT}/{name}.eps')                   # vector, as the guidelines request
    except Exception as e:
        print('  eps skipped for', name, e)
    plt.close(fig)
    print('wrote', name)


def cells(ds):
    p = f'{DS[ds]}/audit_cells.csv'
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------------------------------------------------------- Fig 1: the failure
def fig1():
    fig, ax = plt.subplots(2, 3, figsize=(7.6, 4.4), sharey='row')
    for row, ds in enumerate(['4g', '5g']):
        A = cells(ds)
        if A is None: continue
        U = A[~A.charged]
        for p, g in U[U.temp0 == 0.2].groupby('policy'):
            g = g.sort_values('n')
            ax[row, 0].plot(g.n, g.coverage, 'o-', ms=3.4, lw=1.2, color=COL[p], label=p)
            ax[row, 0].fill_between(g.n, g.cov_lo, g.cov_hi, color=COL[p], alpha=.15, lw=0)
        for p, g in U[U.n == 20000].groupby('policy'):
            g = g.sort_values('temp0')
            ax[row, 1].plot(g.temp0, g.coverage, 'o-', ms=3.4, lw=1.2, color=COL[p])
            ax[row, 1].fill_between(g.temp0, g.cov_lo, g.cov_hi, color=COL[p], alpha=.15, lw=0)
        sc = ax[row, 2].scatter(U.wmax_mean, U.coverage, c=np.log10(U.ess), s=15,
                                cmap='viridis', edgecolor='none')
        for a in ax[row]:
            a.axhline(NOM, color='r', ls='--', lw=.8); a.set_ylim(-.04, 1.06)
        ax[row, 0].set_xscale('log'); ax[row, 2].set_xscale('log')
        ax[row, 0].set_ylabel(f'coverage ({ds.upper()})')
        cb = fig.colorbar(sc, ax=ax[row, 2], pad=.02); cb.set_label(r'$\log_{10}$ ESS', fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    panel_label(ax[0, 0], r'(a) weak exploration, $T_0=0.2$', fontsize=8)
    panel_label(ax[0, 1], r'(b) $n=20{,}000$', fontsize=8)
    panel_label(ax[0, 2], '(c) the tail, not the average', fontsize=8)
    ax[1, 0].set_xlabel('logged decisions $n$'); ax[1, 1].set_xlabel(r'incumbent exploration $T_0$')
    ax[1, 2].set_xlabel(r'mean max importance weight')
    ax[0, 0].legend(fontsize=6.3, loc='upper right')
    save(fig, 'fig1_failure')


# ---------------------------------------------------------------- Fig 2: mechanisms
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.3))
    A4, A5 = cells('4g'), cells('5g')
    for A, mk, lab in [(A4, 'o', '4G'), (A5, 's', '5G')]:
        if A is None: continue
        U = A[~A.charged]
        ax[0].scatter(U.ess/U.n, U.coverage, s=14, marker=mk, alpha=.75, label=lab)
        ch = A[A.charged][['temp0', 'n', 'policy', 'charge']].rename(columns={'charge': 'cbar'})
        U = U.merge(ch, on=['temp0', 'n', 'policy'])
        ax[1].scatter(U.cbar + 1e-4, U.coverage, s=14, marker=mk, alpha=.75, label=lab)
    for a in ax[:2]:
        a.axhline(NOM, color='r', ls='--', lw=.8); a.set_ylim(-.04, 1.06)
    ax[0].set_xlabel('effective sample size / n'); ax[0].set_ylabel('coverage')
    panel_label(ax[0], '(a) ESS does not separate', fontsize=8); ax[0].legend(fontsize=6.5)
    ax[1].set_xscale('log'); ax[1].set_xlabel(r'charge $\bar c$ (+$10^{-4}$)')
    panel_label(ax[1], r'(b) the charge does', fontsize=8)
    A = cells('4g')
    if A is not None:
        R = pd.read_csv(f'{RES}/4g/audit_sweep_raw.csv')
        R = R[~R.charged]
        for p, g in R[R.temp0 == 0.2].groupby('policy'):
            d = []
            for n, gg in g.groupby('n'):
                tc = stats.t.ppf(.95, n-1); se = (gg.d_hat - gg.lb_t)/tc
                d.append((n, se.mean()/gg.d_hat.std(), abs(gg.d_hat.mean()-gg.d_true.mean())/gg.d_hat.std()))
            d = np.array(d)
            ax[2].plot(d[:, 0], d[:, 1], 'o-', ms=3, lw=1.1, color=COL[p])
            ax[2].plot(d[:, 0], d[:, 2], 'o--', ms=3, lw=1.1, color=COL[p], alpha=.55)
        ax[2].axhline(1, color='k', lw=.7); ax[2].set_xscale('log'); ax[2].set_yscale('log')
        ax[2].set_xlabel('logged decisions $n$'); panel_label(ax[2], '(c) two mechanisms', fontsize=8)
        ax[2].legend(handles=[Line2D([], [], ls='-', color='k', label='reported SE / true SD'),
                              Line2D([], [], ls='--', color='k', alpha=.55, label='|bias| / true SD')],
                     fontsize=6)
    save(fig, 'fig2_mechanisms')


# ---------------------------------------------------------------- Fig 3: the repair
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    width = .36
    for j, ds in enumerate(['4g', '5g']):
        A = cells(ds)
        if A is None: continue
        u, c = A[~A.charged], A[A.charged]
        order = sorted(u.policy.unique())
        x = np.arange(len(order))
        wu = [u[u.policy == p].coverage.min() for p in order]
        wc = [c[c.policy == p].coverage.min() for p in order]
        ax[j].bar(x - width/2, wu, width, label='standard', color='#A33B3B')
        ax[j].bar(x + width/2, wc, width, label='with closed-form charge', color='#2E6FAD')
        ax[j].axhline(NOM, color='r', ls='--', lw=.8)
        ax[j].set_xticks(x); ax[j].set_xticklabels(order, rotation=12, fontsize=7)
        ax[j].set_ylim(0, 1.08); ax[j].set_title(f'{ds.upper()} corpus', fontsize=8.5)
    ax[0].set_ylabel('worst-case coverage'); ax[0].legend(fontsize=6.5, loc='upper left')
    save(fig, 'fig3_repair')


# ---------------------------------------------------------------- Fig 4: frontier
def fig4():
    p = f'{RES}/4g/frontier.csv'
    if not os.path.exists(p): return
    F = pd.read_csv(p); V = pd.read_csv(f'{RES}/4g/frontier_validation.csv')
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax[0].scatter(2*F.cbar, F.delta, s=13, c=np.where(F.certifiable, '#4C9F70', '#A33B3B'),
                  edgecolor='none')
    lim = [0, 2*F.cbar.max()*1.02]
    ax[0].plot(lim, lim, 'k-', lw=.9)
    ax[0].set_xlabel(r'$2\,r_{\max}\,\bar c$  (known before any data)')
    ax[0].set_ylabel(r'true gain $\Delta$'); ax[0].set_ylim(-.035, .035)
    panel_label(ax[0], '(a) certifiability frontier', fontsize=8)
    ax[0].legend(handles=[Line2D([], [], marker='o', ls='', color='#4C9F70', label='predicted certifiable'),
                          Line2D([], [], marker='o', ls='', color='#A33B3B', label='predicted not')],
                 fontsize=6.3, loc='lower right')
    ok = (V.predicted == (V.acceptance > 0))
    ax[1].scatter(V.cbar + 1e-4, V.acceptance, s=24, c=np.where(ok, '#4C9F70', '#A33B3B'))
    ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\bar c$'); ax[1].set_ylabel('realised acceptance')
    ax[1].set_title(f'(b) prediction matches the gate in {ok.mean():.0%} of settings', fontsize=8)
    save(fig, 'fig4_frontier')


# ---------------------------------------------------------------- Fig 5: estimator family
def fig5():
    p = f'{RES}/4g/estimator_family_summary.csv'
    if not os.path.exists(p): return
    S = pd.read_csv(p)
    kinds = ['ips', 'snips', 'dr', 'switch', 'shrink']
    x = np.arange(len(kinds)); w = .36
    cov0 = [S[(S.kind == k) & (~S.charged)].coverage.min() for k in kinds]
    cov1 = [S[(S.kind == k) & (S.charged)].coverage.min() for k in kinds]
    g = S[S.d_true > 0]
    pw0 = [g[(g.kind == k) & (~g.charged)].acceptance.mean() for k in kinds]
    pw1 = [g[(g.kind == k) & (g.charged)].acceptance.mean() for k in kinds]
    ch = [S[(S.kind == k) & (S.charged)].charge.mean() for k in kinds]
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.3))
    ax[0].bar(x - w/2, cov0, w, color='#A33B3B', label='as published')
    ax[0].bar(x + w/2, cov1, w, color='#2E6FAD', label='+ generalised charge')
    ax[0].axhline(NOM, color='r', ls='--', lw=.8); ax[0].set_ylabel('worst-case coverage')
    ax[0].legend(fontsize=6.3)
    ax[1].bar(x - w/2, pw0, w, color='#A33B3B'); ax[1].bar(x + w/2, pw1, w, color='#2E6FAD')
    ax[1].set_ylabel('power')
    ax[2].bar(x, ch, .6, color='#4C9F70'); ax[2].set_ylabel(r'mean charge $\bar c$')
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(kinds, rotation=15, fontsize=7)
    ax[2].set_title('the charge ranks the transforms', fontsize=8)
    save(fig, 'fig5_estimator_family')


# ---------------------------------------------------------------- Fig 6: transfer
def fig6():
    p = f'{RES}/4g/transfer_summary.csv'
    if not os.path.exists(p): return
    B = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for j, ch in enumerate([False, True]):
        piv = B[B.charged == ch].pivot_table(index='src', columns='dst', values='coverage')
        im = ax[j].imshow(piv.values, vmin=0, vmax=1, cmap='RdYlGn')
        ax[j].set_xticks(range(len(piv.columns))); ax[j].set_xticklabels(piv.columns, rotation=45, fontsize=6.5)
        ax[j].set_yticks(range(len(piv.index))); ax[j].set_yticklabels(piv.index, fontsize=6.5)
        ax[j].grid(False); ax[j].set_xlabel('deployment regime')
        for r in range(piv.shape[0]):
            for c_ in range(piv.shape[1]):
                ax[j].text(c_, r, f'{piv.values[r, c_]:.2f}', ha='center', va='center', fontsize=6)
        ax[j].set_title('(b) with the charge' if ch else '(a) standard certificate', fontsize=8)
    ax[0].set_ylabel('logging regime')
    fig.colorbar(im, ax=ax, pad=.02, shrink=.85).set_label('coverage vs the deployment regime', fontsize=7)
    save(fig, 'fig6_transfer')


# ---------------------------------------------------------------- Fig 7: sequential
def fig7():
    p = f'{RES}/4g/sequential_summary.csv'
    if not os.path.exists(p): return
    S = pd.read_csv(p)
    sub = S[(S.temp0 == 1.2) & (S.H == S.H.max())]
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.3))
    for pol, g in sub.groupby('policy'):
        g = g.sort_values('k')
        ax[0].plot(g.k, g.bias, 'o-', ms=3.4, lw=1.2, color=COL[pol], label=pol)
        ax[1].plot(g.k, g.coverage, 'o-', ms=3.4, lw=1.2, color=COL[pol])
        ax[2].plot(g.k, g.acceptance, 'o-', ms=3.4, lw=1.2, color=COL[pol])
    ax[0].axhline(0, color='k', lw=.7); ax[0].set_ylabel('bias'); panel_label(ax[0], '(a) bias', fontsize=8)
    ax[1].axhline(NOM, color='r', ls='--', lw=.8); ax[1].set_ylabel('coverage')
    panel_label(ax[1], '(b) validity', fontsize=8)
    ax[2].set_ylabel('acceptance'); panel_label(ax[2], '(c) power', fontsize=8); ax[2].legend(fontsize=6.3)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('window $k$')
        a.set_xticks(sorted(sub.k.unique())); a.set_xticklabels(sorted(sub.k.unique()))
    save(fig, 'fig7_sequential')


# ---------------------------------------------------------------- Fig 8: anytime
def fig8():
    p = f'{RES}/4g/anytime_summary.csv'
    if not os.path.exists(p): return
    S = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    w = .36; Ms = sorted(S.M.unique()); x = np.arange(len(Ms))
    ax[0].bar(x - w/2, [S[S.M == m].cov_t_peek.min() for m in Ms], w, color='#A33B3B',
              label='fixed-$n$ bound, monitored')
    ax[0].bar(x + w/2, [S[S.M == m].cov_cs_peek.min() for m in Ms], w, color='#2E6FAD',
              label='confidence sequence')
    ax[0].axhline(NOM, color='r', ls='--', lw=.8); ax[0].set_ylabel('worst-case coverage')
    ax[0].legend(fontsize=6.3)
    g = S[S.d_true > 0]
    ax[1].bar(x - w/2, [g[g.M == m].acc_t.mean() for m in Ms], w, color='#A33B3B')
    ax[1].bar(x + w/2, [g[g.M == m].acc_cs.mean() for m in Ms], w, color='#2E6FAD')
    ax[1].set_ylabel('power')
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([f'M={m:g}' for m in Ms], fontsize=7)
    save(fig, 'fig8_anytime')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f()
        except Exception as e:
            print(f'{f.__name__}: skipped ({e})')
    print('\n'.join(sorted(os.listdir(OUT))))
