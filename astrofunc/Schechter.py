# -*- coding: utf-8 -*-
# @Author: ssli
# @Date:   2023-03-17 12:53:44
# @Last Modified by:   ssli
# @Last Modified time: 2023-03-17 17:03:15

# Schechter function and its variations
## https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S/abstract

import numpy as np 
import pandas as pd 

__all__ = ['SchechterSimple', 'SchechterDouble']

def SchechterSimple(m_list=None, f_list=None,
                        phi_star=0.009, 
                        alpha=-1.26, 
                        m_star=-20.73, 
                        f_star=None,  
                        magORflux='mag'):
    r"""
    Schechter luminosity function 

    the default values are taken from Loveday et al. 2012
        for All in r-band (Table 3)

    Parameters
    ----------
    m_list : numpy array (default: None)
        A list of magnitude (-5log(h)) for which the function values are calculated (for magORflux='mag').

    f_list : numpy array (default: None)
        A list of flux for which the function values are calculated (for magORflux='flux').

    phi_star : float (default: 0.009)
        The normalization factor in units of number density.

    alpha : float (default: -1.26)
        The power law index, also known as the faint-end slope. 

    m_star : float (default: -20.73)
        The characteristic magnitude where the power-law form of the
        function cuts off (for magORflux='mag').

    f_star : float (default: None)
        The characteristic flux where the power-law form of the
        function cuts off (for magORflux='flux').

    magORflux : string (default: 'mag')
        Working on magnitude ('mag') or flux ('flux')

    Notes
    -----
    Model formula in magnitude (with :math:`\phi^{*}` for ``phi_star``, :math:`M^{*}`
    for ``m_star``, and :math:`\alpha` for ``alpha``):

    .. math::

        \phi(M) = (0.4 \ln 10) \ \phi^{*} \
            [{10^{0.4 (M^{*} - M)}}]^{\alpha + 1} \
            \exp{[-10^{0.4 (M^{*} - M)}]}

    """

    if magORflux.lower() == 'mag':
        if m_list is None:
            raise Exception('Please provide m_list at least!')
        factor = 10 ** (-0.4 * (m_list - m_star))
        return 0.4 * np.log(10) * phi_star * factor ** (alpha + 1) * np.exp(-factor)
    elif magORflux.lower() == 'flux':
        raise Exception('magORflux = flux not supported yet!')
    else:
        raise Exception(f'Unrecognised {magORflux} for magORflux!')


def SchechterDouble(m_list=None, f_list=None,
                        phi_star=0.0102, 
                        alpha=0.14, beta=-1.47,
                        m_star=-19.92, m_t=-19.86, 
                        f_star=None, f_t=None,  
                        magORflux='mag'):
    r"""
    Schechter luminosity function with double-power law

    the default values are taken from Loveday et al. 2012
        for All in r-band (Table 4)

    Parameters
    ----------
    m_list : numpy array (default: None)
        A list of magnitude (-5log(h)) for which the function values are calculated (for magORflux='mag').

    f_list : numpy array (default: None)
        A list of flux for which the function values are calculated (for magORflux='flux').

    phi_star : float (default: 0.0102)
        The normalization factor in units of number density.

    alpha : float (default: 0.14)
        The power law index, also known as the faint-end slope. 

    beta : float (default: -1.47)
        The power law index in the second power law function. 

    m_star : float (default: -19.92)
        The characteristic magnitude where the power-law form of the
        function cuts off (for magORflux='mag').

    m_t : float (default: -19.86)
        The second characteristic magnitude (for magORflux='mag').

    f_star : float (default: None)
        The characteristic flux where the power-law form of the
        function cuts off (for magORflux='flux').

    f_t : float (default: None)
        The second characteristic flux (for magORflux='flux').

    magORflux : string (default: 'mag')
        Working on magnitude ('mag') or flux ('flux')

    Notes
    -----
    Model formula in magnitude (with :math:`\phi^{*}` for ``phi_star``, :math:`M^{*}`
    for ``m_star``, :math:`M_{t}` for ``m_t``,
    :math:`\alpha` for ``alpha`` and :math:`\beta` for ``beta``):

    .. math::

        \phi(M) = (0.4 \ln 10) \ \phi^{*} \
            [{10^{0.4 (M^{*} - M)}}]^{\alpha + 1} \
            \exp{[-10^{0.4 (M^{*} - M)}]} \
            [[{10^{0.4 (M_{t} - M)}}]^{\beta} + 1]

    """

    if magORflux.lower() == 'mag':
        if m_list is None:
            raise Exception('Please provide m_list at least!')
        factor = 10 ** (-0.4 * (m_list - m_star))
        factor_t = 10 ** (-0.4 * (m_list - m_t))
        return 0.4 * np.log(10) * phi_star * factor ** (alpha + 1) * np.exp(-factor) * (factor_t ** beta + 1)
    elif magORflux.lower() == 'flux':
        raise Exception('magORflux = flux not supported yet!')
    else:
        raise Exception(f'Unrecognised {magORflux} for magORflux!')

# test the code
if __name__ == '__main__':
    import plotting

    m_list = np.arange(-23, -12, 0.1)

    # simple Schechter
    phi_simple = SchechterSimple(m_list=m_list, f_list=None,
                            phi_star=0.009, alpha=-1.26, 
                            m_star=-20.73, f_star=None,  
                            magORflux='mag')

    # double-power-law Schechter
    phi_double = SchechterDouble(m_list=m_list, f_list=None,
                        phi_star=0.0102, alpha=0.14, beta=-1.47,
                        m_star=-19.92, m_t=-19.86, f_star=None, f_t=None,  
                        magORflux='mag')

    plotting.LinePlotFunc(outpath='show',
                xvals=[m_list, m_list], yvals=[phi_simple, phi_double],
                COLORs=['k', 'k'], LABELs=None, LINEs=['dotted', '-'], LINEWs=None, POINTs=['', ''], POINTSs=None, fillstyles=None,
                XRANGE=None, YRANGE=None,
                XLABEL=None, YLABEL=None, TITLE=None,
                xtick_min_label=True, xtick_spe=None, ytick_min_label=True, ytick_spe=None,
                vlines=None, vline_styles=None, vline_colors=None, vline_labels=None, vline_widths=None,
                hlines=None, hline_styles=None, hline_colors=None, hline_labels=None, hline_widths=None,
                xlog=False, invertX=False, ylog=True, invertY=False, 
                loc_legend='best', legend_frame=False,
                font_size=12, usetex=False,
                FIGSIZE=[6.4, 4.8],
                texPacks=None)