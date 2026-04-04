"""Data loading tools.

Loads data from codes into a dictionary, with units and array structures
standardised. Also adds a ``meta`` subdictionary, which contains units,
array dimensions and the data source.
"""

#Functions
#---------
#
#    amset:
#        from the amset transport json.
#    amset_mesh:
#        from the amset mesh h5.
#    boltztrap:
#        from the tp boltztrap hdf5.
#    boltztrap2:
#        from the tp boltztrap2 hdf5.
#    phono3py:
#        from the phono3py kappa hdf5.
#    phonopy_dispersion:
#        from the phonopy or sumo band.yaml.
#    phonopy_dos:
#        from the phonopy projected_dos.dat.
#    shengbte:
#        from a ShengBTE/FourPhonon calculation directory.
#
#    get_path:
#        gets high path from phonopy dispersion data.
#"""

import numpy as np
import tp
from tp import settings

def amset(filename, quantities='all', doping='n'):
    """Loads AMSET transport data from json.

    Includes unit conversion and outputs units (see tp.settings).
    Swaps temperature and doping indices so temperature is first, for
    consistency with other codes, and mobility is formatted as an array,
    like scattering_rates is in the mesh.h5 file. Maintains basic
    compatibility with amset 0.1.

    Arguments
    ---------

        filename : str
            amset transport json filepath.

        quantites : str or list, optional
            values to extract. Accepts AMSET keys and power_factor.
            All loads the whole file, power factor can be added if it
            is not in the file. Loads dependent properties.
            Default: all.
        doping : str, optional
            doing type (n or p). If there is more than one, defaults to
            n, else this is ignored.

    Returns
    -------

        dict
            extracted values.
    """

    import json

    # name conversions and abbreviations

    aconversions = settings.amset_conversions()
    conversions = settings.conversions()
    anames = settings.to_amset()
    tnames = settings.to_tp()
    units = settings.units()
    dimensions = settings.dimensions()
    if isinstance(quantities, str): quantities = quantities.split()
    quantities = [anames[q] if q in anames else q for q in quantities]

    # quantities derived from those in the file
    derived = {'power_factor': ['conductivity', 'seebeck']}

    # add dependent variables
    for d in ['doping', 'temperatures']:
        if d not in quantities:
            for q in quantities:
                if q in tnames:
                    q = tnames[q]
                if q in dimensions and \
                   (d in dimensions[q] or (d in tnames and tnames[d] in dimensions[q])):
                    quantities.append(d)
                    break
    for q in derived:
        if q in quantities:
            for q2 in derived[q]:
                if q2 not in quantities:
                    quantities.append(q2)

    # load data

    with open(filename) as f:
        data = json.load(f)
    if 'all' in quantities:
        if 'power_factor' in quantities:
            quantities = [*data.keys(), 'power_factor']
        else:
            quantities = list(data.keys())

    if 'doping' in quantities:
        if 'data' in data['doping']:
            d = np.array(data['doping']['data'])
        else:
            d = np.array(data['doping'])
        if (d < 0).all():
            doping = 'n'
            di = np.where(d < 0)[0]
        elif (d > 0).all():
            doping = 'p'
            di = np.where(d > 0)[0]
        elif doping == 'n':
            di = np.where(d < 0)[0]
        elif doping == 'p':
            di = np.where(d > 0)[0]

    data2 = {'meta': {'doping_type':       doping,
                      'electronic_source': 'amset',
                      'units':             {},
                      'dimensions':        {}}}
    for q in quantities:
        if q in derived:
            continue
        qs = [*list(data), *list(derived)]
        assert q in data, \
               '{} unrecognised. Quantity must be in {} or {}.'.format(q,
               ', '.join(qs[:-1]), qs[-1])
        q2 = tnames[q] if q in tnames else q
        # compatibility with previous version
        if isinstance(data[q], dict) and 'data' in data[q]:
            data2[q2] = data[q]['data']
        else:
            data2[q2] = data[q]
        if q2 in dimensions and 'doping' in dimensions[q2]:
            # temperature index first for consistency with other codes
            # With the latest version of resolve, this is unneccessary 
            # Should it be removed?
            if 'stype' in dimensions[q2]:
                for t in data2[q2]:
                    data2[q2][t] = np.array(data2[q2][t])[di]
                    if 'temperature' in dimensions[q2]:
                        data2[q2][t] = np.swapaxes(data2[q2][t],0,1)
            else:
                data2[q2] = np.array(data2[q2])[di]
                if 'temperature' in dimensions[q2]:
                    data2[q2] = np.swapaxes(data2[q2],0,1)
        if q2 in dimensions and 'stype' in dimensions[q2]:
            if q2 == 'mobility':
                data[q2]['Total'] = data[q2].pop('overall')
            if 'stype' not in data2:
                data2['stype'] = list(data[q].keys())
            # for consistency with the format in the mesh data
            data2[q2] = [data2[q2][l] for l in data2['stype']]
        if q2 in units:
            data2['meta']['units'][q2] = units[q2]
        if q2 in dimensions:
            data2['meta']['dimensions'][q2] = dimensions[q2]

    for c in aconversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(aconversions[c]))

    if 'power_factor' in quantities:
        data2 = tp.calculate.power_factor_fromdict(data2, use_tprc=False)

    for c in conversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(conversions[c]))

    return data2

def amset_mesh(filename, quantities='all', doping='n', spin='avg'):
    """Loads AMSET mesh data from h5.

    Can also weight rates. Includes unit conversion and outputs units
    (see tp.settings). Swaps temperature and doping indices so
    temperature is first, for consistency with other codes.

    Arguments
    ---------

        filename : str
            amset mesh h5 filepath.

        quantites : str or list, optional
            values to extract. Accepts AMSET keys, without spin
            channels, which are dealt with in the spin variable.
            Also accepts ibz_weights, the weights of the irreducible
            k-points, fd_weights, the weights of the energies wrt the
            derivative of the Fermi-Dirac distribution, weighted_rates,
            scattering_rates weighted by fd_weights and averaged over
            kpoints and occupation, the Fermi-Dirac occupation.
            "all" loads all quantities in the file, which does not
            include ibz_weights, fd_weights, weighted_rates or
            occupation. Loads dependent properties. Default: all.

        doping : str, optional
            doing type (n or p). If there is more than one, defaults to
            n, else this is ignored.

        spin : str, optional
            spin. Accepts up, down or avg. If avg and there is only one
            spin channel, selects that, else averages both. If avg,
            assumes non-spin separated bands and so multiplies occupation
            by 2. Default: avg.

    Returns
    -------

        dict
            extracted values.
    """

    import h5py

    # name conversions and abbreviations

    aconversions = settings.amset_conversions()
    conversions = settings.conversions()
    anames = settings.to_amset()
    tnames = settings.to_tp()
    units = settings.units()
    dimensions = settings.dimensions()
    dimensions['occupation'] = ['temperature', 'doping', 'kpoint', 'band']
    if isinstance(quantities, str): quantities = quantities.split()
    quantities = [anames[q] if q in anames else q for q in quantities]

    # list of abbriviations and dependent quantites
    subs = {'weights':        ['ibz_weights', 'fd_weights'],
            'weighted_rates': ['scattering_rates', 'energies', 'fermi_levels', 'weighted_rates'],
            'mean_free_path': ['scattering_rates', 'velocities', 'mean_free_path'],
            'weighted_mfp':   ['scattering_rates', 'velocities', 'mean_free_path', 'energies', 'fermi_levels', 'weighted_mfp'],
            'occupation':     ['energies', 'fermi_levels', 'occupation']}
    hasspin = ['energies', 'vb_idx', 'scattering_rates', 'velocities']

    l = len(quantities)
    i = 0
    while i < l:
        if quantities[i] in subs:
            quantities.extend(subs[quantities[i]])
            del quantities[i]
            l -= 1
        else:
            i += 1

    # add dependent variables

    for d in ['doping', 'ir_kpoints', 'temperatures', 'scattering_labels']:
        if d not in quantities:
            for q in quantities:
                if q in tnames:
                    q = tnames[q]
                if q in dimensions and \
                   (d in dimensions[q] or (d in tnames and tnames[d] in dimensions[q])):
                    quantities.append(d)
                    break

    if 'temperatures' in quantities:
        quantities.insert(0, quantities.pop(quantities.index('temperatures')))

    # load data

    def resolve_spin(data, q, spin):
        """Resolves spin in AMSET.

        Either returns the up or down value, or averages them.

        Arguments
        ---------

        data : dict
            data dictionary from AMSET.
        q : str
            quantity to resolve
        spin : str
            spin to use. Must be up, down or average.

        Returns
        -------

        array
            resolved data
        """

        if spin in ['avg', 'average']:
            resolved = np.average([data['{}_up'.format(q)][()],
                                   data['{}_down'.format(q)][()]], axis=0)
        elif spin in ['up', 'down']:
            resolved = data['{}_{}'.format(q, spin)][()]
        else:
            raise Exception('spin must be up or down or average')

        return resolved

    with h5py.File(filename, 'r') as f:
        if 'all' in quantities:
            q2 = quantities
            quantities = list(f.keys())
            for q in ['ibz_weights', 'fd_weights', 'weighted_rates', 'occupation']:
                if q in q2:
                    quantities.append(q)
        if 'doping' in quantities:
            d = np.array(f['doping'][()])
            if (d < 0).all():
                doping = 'n'
                di = np.where(d < 0)[0]
            elif (d > 0).all():
                doping = 'p'
                di = np.where(d > 0)[0]
            elif doping == 'n':
                di = np.where(d < 0)[0]
            elif doping == 'p':
                di = np.where(d > 0)[0]

        # spin2 so if avg selected doubles occupation
        if spin in ['avg', 'average']:
            if 'energies_up' in f and 'energies_down' in f:
                spin2 = 'avg'
            elif 'energies_up' in f:
                spin2 = 'up'
            elif 'energies_down' in f:
                spin2 = 'down'
        else:
            spin2 = spin

        data = {'meta': {'doping_type':       doping,
                         'electronic_source': 'amset',
                         'spin':              spin2,
                         'units':             {},
                         'dimensions':        {}}}
        for q in quantities:
            q2 = tnames[q] if q in tnames else q
            if q in hasspin:
                data[q2] = resolve_spin(f, q, spin2)
            elif q in f:
                data[q2] = f[q][()]
            elif q not in ['ibz_weights', 'fd_weights', 'weighted_rates',
                           'mean_free_path', 'weighted_mfp', 'occupation']:
                raise Exception('{} unrecognised. Quantity must be {}, '
                                'ibz_weights, fd_weights, weighted_rates, '
                                'mean_free_path or weighted_mfp or occupation.'
                                ''.format(q, ', '.join(f)))
            if q == 'scattering_rates':
                data[q2] = [*list(data[q2]), list(np.sum(data[q2], axis=0))]
            if q in ['ibz_weights', 'weighted_rates', 'weighted_mfp']:
                _, data['ibz_weights'] = np.unique(f['ir_to_full_kpoint_mapping'],
                                                   return_counts=True)
            if q in ['fd_weights', 'weighted_rates', 'weighted_mfp']:
                e = resolve_spin(f, 'energies', spin2)
                data['fd_weights'] = -tp.calculate.dfdde(e, f['fermi_levels'],
                                                         f['temperatures'],
                                                         amset_order=True)
            if q in ['mean_free_path', 'weighted_mfp']:
                data['velocities'] = np.array(data['velocities'])
                data['scattering_rates'] = np.array(data['scattering_rates'])
                data['mean_free_path'] = data['velocities'][None, None, None, :, :, :] / \
                                         data['scattering_rates'][:, :, :, :, :, None]
            if q in ['weighted_rates', 'weighted_mfp']:
                data['total_weights'] = data['fd_weights'] * data['ibz_weights']
                data['normalised_weights'] = data['total_weights'] / \
                                             np.sum(data['total_weights'],
                                                    axis=(2,3))[:,:,None,None]
            if q =='weighted_rates':
                from copy import deepcopy
                rates = np.array(deepcopy(data['scattering_rates']))
                rates[rates > 1e20] = 1e15
                data[q2] = rates * data['normalised_weights']
                data[q2] = data[q2].sum(axis=(3,4))
            if q == 'weighted_mfp':
                data[q2] = data['mean_free_path'] * \
                           data['normalised_weights'][None, :, :, :, :, None]
                data[q2] = data[q2].sum(axis=(3,4))
            if q == 'occupation':
                data['occupation'] = tp.calculate.fd_occupation(
                                     data['energy'], data['temperature'],
                                     data['fermi_level'])
                if spin in ['avg', 'average']:
                    data['occupation'] *= 2

        for q2 in data:
            q = anames[q2] if q2 in anames else q2
            if q2 in dimensions and 'doping' in dimensions[q2]:
                # temperature in first index for consistency with other codes
                if 'stype' in dimensions[q2]:
                    if 'temperature' in dimensions[q2]:
                        data[q2] = np.swapaxes(data[q2],1,2)
                        data[q2] = np.array(data[q2])[:,:,di]
                    else:
                        data[q2] = np.array(data[q2])[:,di]
                else:
                    if 'temperature' in dimensions[q2]:
                        data[q2] = np.swapaxes(data[q2],0,1)
                        data[q2] = np.array(data[q2])[:,di]
                    else:
                        data[q2] = np.array(data[q2])[di]
            if q2 in units:
                data['meta']['units'][q2] = units[q2]
            if q2 in dimensions:
                data['meta']['dimensions'][q2] = dimensions[q2]

    if 'stype' in data:
        data['stype'] = [l.decode('UTF-8') for l in data['stype']]
        data['stype'].append('Total')

    for c in aconversions:
        if c in data:
            data[c] = np.multiply(data[c], float(aconversions[c]))

    for c in conversions:
        if c in data:
            data[c] = np.multiply(data[c], float(conversions[c]))

    return data

def boltztrap(filename, quantities='all', doping='n'):
    """Loads BoltzTraP data from the tp boltztrap.hdf5 file.

    Includes unit conversion and outputs units (see tp.settings).

    Arguments
    ---------

        filename : str
            boltztrap.hdf5 filepath.

        quantites : str or list, optional
            values to extract. Accepts boltztrap.hdf5 keys or all.
            Default: all.

        doping : str, optional
            doping.  Default: n.

    Returns
    -------

        dict
            extracted values.
    """

    import h5py

    # name conversions and abbreviations

    assert doping in ['n', 'p'], 'doping must be n or p'

    bconversions = settings.boltztrap_conversions()
    conversions = settings.conversions()
    bnames = settings.to_boltztrap()
    tnames = settings.to_tp()
    units = settings.units()
    dimensions = settings.boltztrap_dimensions()
    if isinstance(quantities, str): quantities = quantities.split()
    if 'all' not in quantities:
        quantities = [bnames[q] if q in bnames else q for q in quantities]

        # add dependent variables

        for d in ['doping', 'temperature']:
            if d not in quantities:
                for q in quantities:
                    if q in tnames:
                        q = tnames[q]
                    if q in dimensions and \
                       (d in dimensions[q] or (d in tnames and tnames[d] in dimensions[q])):
                        quantities.append(d)
                        break

    # load data

    with h5py.File(filename, 'r') as f:
        if 'all' in quantities:
            quantities = list(f.keys())
            quantities.remove('meta')
        data = {'meta': {'doping_type':       doping,
                         'electronic_source': 'boltztrap',
                         'units':             {},
                         'dimensions':        {}}}
        for q in quantities:
            assert q in f, '{} unrecognised. Quantity must be in {} or {}.'.format(
                            q, ', '.join(list(f)[:-1]), list(f)[-1])
            q2 = tnames[q] if q in tnames else q
            if q2 in dimensions and 'dtype' in dimensions[q2]:
                data[q2] = f[q][doping][()]
                dimensions[q2].remove('dtype')
            else:
                data[q2] = f[q][()]
            if q2 in units:
                data['meta']['units'][q2] = units[q2]
            if q2 in dimensions:
                data['meta']['dimensions'][q2] = dimensions[q2]

    for c in bconversions:
        if c in data:
            data[c] = np.multiply(data[c], float(bconversions[c]))

    for c in conversions:
        if c in data:
            data[c] = np.multiply(data[c], float(conversions[c]))

    return data

def boltztrap2(filename, quantities='all'):
    """Loads BoltzTraP2 data from the tp tp-btp2.hdf5 file.

    Includes unit conversion and outputs units (see tp.settings).

    Arguments
    ---------

        filename : str
            tp-btp2.hdf5 filepath.

        quantites : str or list, optional
            values to extract. Accepts boltztrap.hdf5 keys or all.
            Default: all.

    Returns
    -------

        dict
            extracted values.
    """

    import h5py

    # name conversions and abbreviations

    bconversions = settings.boltztrap2_conversions()
    conversions = settings.conversions()
    bnames = settings.to_boltztrap2()
    tnames = settings.to_tp()
    units = settings.units()
    dimensions = settings.dimensions()
    if isinstance(quantities, str): quantities = quantities.split()
    if 'all' not in quantities:
        quantities = [bnames[q] if q in bnames else q for q in quantities]

        # add dependent variables

        for d in ['doping', 'temperature']:
            if d not in quantities:
                for q in quantities:
                    if q in tnames:
                        q = tnames[q]
                    if q in dimensions and \
                       (d in dimensions[q] or (d in tnames and tnames[d] in dimensions[q])):
                        quantities.append(d)
                        break

    # load data

    with h5py.File(filename, 'r') as f:
        if 'all' in quantities:
            quantities = list(f.keys())
            quantities.remove('meta')
        data = {'meta': {'electronic_source': 'boltztrap2',
                         'units':             {},
                         'dimensions':        {}}}
        for q in quantities:
            assert q in f, '{} unrecognised. Quantity must be in {} or {}.'.format(
                            q, ', '.join(list(f)[:-1]), list(f)[-1])
            q2 = tnames[q] if q in tnames else q
            data[q2] = f[q][()]
            if q2 in units:
                data['meta']['units'][q2] = units[q2]
            if q2 in dimensions:
                data['meta']['dimensions'][q2] = dimensions[q2]

    for c in bconversions:
        if c in data:
            data[c] = np.multiply(data[c], float(bconversions[c]))

    for c in conversions:
        if c in data:
            data[c] = np.multiply(data[c], float(conversions[c]))

    return data

def phono3py(filename, quantities='all'):
    """Loads Phono3py data from kappa hdf5.

    Can also calculate lifetimes, mean free paths and occupations.
    Includes unit conversions and outputs units and index order for all
    the data (see tp.settings). Also corrects mode_kappa for different
    phono3py versions. Also converts the default 6x1 direction matrices
    into 3x3 ones for compatability with other codes.

    Arguments
    ---------

        filename : str
            phono3py kappa hdf5 filepath.

        quantities : str or list, optional
            values to extract. Accepts Phono3py keys, lifetime,
            mean_free_path and occupation. 'all' loads all Phono3py
            keys, but lifetime etc. must be loaded separately.
            Default: all.

    Returns
    -------

        dict
            output data.
    """

    import h5py

    # name conversions and abbreviations

    pconversions = settings.phono3py_conversions()
    conversions = settings.conversions()
    pnames = settings.to_phono3py()
    tnames = settings.to_tp()
    units = settings.units()
    dimensions = settings.dimensions()
    if isinstance(quantities, str): quantities = quantities.split()
    quantities = [pnames[q] if q in pnames else q for q in quantities]
    subs = {'dispersion': ['qpoint'],
            'waterfall':  ['frequency'],
            'wideband':   ['frequency', 'gamma', 'qpoint']}

    # quantities derived from those in the file
    derived = {'lifetime': ['gamma'],
               'mean_free_path': ['gamma', 'group_velocity'],
               'occupation': ['frequency']}

    for i in range(len(quantities)):
        while True:
            if quantities[i] in subs:
                quantities.extend(subs[quantities[i]])
                del quantities[i]
            else:
                break
        
    for q in derived:
        if q in quantities:
            for q2 in derived[q]:
                if q2 not in quantities:
                    quantities.append(q2)

    quantities = list(np.ravel(quantities))

    # add dependent variables

    for d in ['temperature', 'qpoint']:
        if d not in quantities:
            for q in quantities:
                if q in tnames:
                    q = tnames[q]
                if q in dimensions and d in dimensions[q]:
                    quantities.append(d)
                    break

    # load and calculate data

    with h5py.File(filename, 'r') as f:
        if 'all' in quantities:
            q2 = quantities
            quantities = list(f.keys())
            for q in derived:
                if q in q2:
                    quantities.append(q)
        data = {'meta': {'kappa_source': 'phono3py',
                         'units':        {},
                         'dimensions':   {}}}
        for q in quantities:
            q2 = tnames[q] if q in tnames else q
            if q2 in units:
                data['meta']['units'][q2] = units[q2]
            if q2 in dimensions:
                data['meta']['dimensions'][q2] = dimensions[q2]
            if q in derived:
                continue
            qs = [*list(f), *list(derived)]
            assert q in f, \
                   '{} unrecognised. Quantity must be in {} or {}.'.format(q,
                   ', '.join(qs[:-1]), qs[-1])
            data[q2] = f[q][()]
            if q2 in dimensions:
                for i, d in enumerate(dimensions[q2]):
                    if d == 6:
                        data[q2] = np.moveaxis(data[q2], i, 0)
                        data[q2] = [[data[q2][0], data[q2][5], data[q2][4]],
                                    [data[q2][5], data[q2][1], data[q2][3]],
                                    [data[q2][4], data[q2][3], data[q2][2]]]
                        data[q2] = np.moveaxis(data[q2], 0, i+1)
                        data[q2] = np.moveaxis(data[q2], 0, i+1)
                        dimensions[q2][i] = 3
                        dimensions[q2].insert(i, 3)

        # check mode_kappa and correct for certain phono3py versions
        if 'mode_kappa' in data:
            try:
                k = round(f['kappa'][()][-1][0], 3)
                mk = round(f['mode_kappa'][()][-1][:,:,0].sum(axis=1).sum(), 3)
                if k != mk:
                    raise ValueError('The sum of mode_kappa does not equal kappa.\n '
                                     'kappa={:.3f}; sum(mode_kappa)={:.3f}.'.format(
                                                                             k, mk))
            except ValueError:
                mk2 = np.divide(f['mode_kappa'][()], np.prod(f['mesh'][()][:]))
                k = round(f['kappa'][()][-1][0], 3)
                mk = round(mk2[-1][:,:,0].sum(axis=1).sum(), 3)
                if k != mk:
                    raise ValueError('Mode kappa has been divided by the mesh, but '
                                     'the sum of mode_kappa does not equal kappa.\n '
                                     'kappa={:.3f}; sum(mode_kappa)={:.3f}.'.format(
                                                                             k, mk))
                else:
                    data['mode_kappa'] = np.divide(data['mode_kappa'],
                                                   np.prod(f['mesh'][()][:]))

    for c in pconversions:
        if c in data:
            data[c] = np.multiply(data[c], float(pconversions[c]))

    if 'lifetime' in quantities or 'mean_free_path' in quantities:
        data['lifetime'] = tp.calculate.lifetime(data['gamma'], use_tprc=False)
        if 'mean_free_path' in quantities:
            data['mean_free_path'] = tp.calculate.mfp(data['gamma'],
                                                      data['group_velocity'],
                                                      use_tprc=False)
    if 'occupation' in quantities:
        data['occupation'] = tp.calculate.be_occupation(data['frequency'],
                                                        data['temperature'],
                                                        use_tprc=False)

    for c in conversions:
        if c in data:
            data[c] = np.multiply(data[c], float(conversions[c]))

    return data

def shengbte(directory, quantities='all'):
    """Loads ShengBTE/FourPhonon data from a calculation directory.

    Reads text-file outputs from ShengBTE and its FourPhonon extension.
    Converts scattering rates to gamma (imaginary self-energy) for
    compatibility with tp plotting functions. Can also calculate
    lifetimes, mean free paths, occupations, and mode-resolved thermal
    conductivity. Includes unit conversions and outputs units and index
    order for all data (see tp.settings).

    The mode_kappa quantity is synthesised from the RTA kinetic theory
    expression: kappa_ij = (1/NV) * C * v_i * v_j * tau, where C is the
    per-mode specific heat from the Bose-Einstein distribution. This is
    exact for RTA but approximate for the iterative BTE solution.

    Optionally loads pre-computed cumulative kappa vs MFP and vs
    frequency as ``cumulative_kappa_mfp`` and
    ``cumulative_kappa_frequency``.

    If FourPhonon outputs are present (BTE.w_3ph, BTE.w_4ph), they can
    be loaded as gamma_3ph and gamma_4ph.

    Arguments
    ---------

        directory : str
            path to the ShengBTE calculation directory containing
            CONTROL, BTE.* files and T*K/ subdirectories.

        quantities : str or list, optional
            values to extract. Accepts tp names, abbreviations or 'all'.
            'all' loads all available quantities except gamma_3ph,
            gamma_4ph, cumulative_kappa_mfp and cumulative_kappa_frequency
            which must be requested explicitly. Default: all.

    Returns
    -------

        dict
            extracted values.
    """

    import math
    import os
    from pathlib import Path
    from scipy.constants import physical_constants

    import f90nml

    # name conversions and abbreviations

    sconversions = settings.shengbte_conversions()
    conversions = settings.conversions()
    tnames = settings.to_tp()
    units = settings.units()
    dims = settings.dimensions()
    if isinstance(quantities, str): quantities = quantities.split()
    quantities = [tnames[q] if q in tnames else q for q in quantities]
    subs = {'dispersion': ['qpoint'],
            'waterfall':  ['frequency'],
            'wideband':   ['frequency', 'gamma', 'qpoint']}

    # quantities derived from those in the file
    derived = {'gamma':           [],
               'gamma_3ph':       [],
               'gamma_4ph':       [],
               'lifetime':        ['gamma'],
               'mean_free_path':  ['gamma', 'group_velocity'],
               'mode_kappa':      ['gamma', 'group_velocity', 'frequency'],
               'occupation':      ['frequency']}

    for i in range(len(quantities)):
        while True:
            if quantities[i] in subs:
                quantities.extend(subs[quantities[i]])
                del quantities[i]
            else:
                break

    for q in derived:
        if q in quantities:
            for q2 in derived[q]:
                if q2 not in quantities:
                    quantities.append(q2)

    quantities = list(np.ravel(quantities))

    # add dependent variables

    for d in ['temperature', 'qpoint']:
        if d not in quantities:
            for q in quantities:
                if q in tnames:
                    q = tnames[q]
                if q in dims and d in dims[q]:
                    quantities.append(d)
                    break

    # parse CONTROL file

    directory = Path(directory).resolve()
    control_path = directory / 'CONTROL'
    raw_control = f90nml.read(control_path).todict()
    control = {}
    for section in raw_control.values():
        control.update(section)
    control.pop('_start_index', None)

    natoms = control['natoms']
    nbands = natoms * 3
    ngrid = np.array(control['ngrid'], dtype=int)
    N = np.prod(ngrid)

    lfactor = control.get('lfactor', 1.0)
    lattvec = np.array(control['lattvec'], dtype=float) * lfactor
    volume = np.abs(np.linalg.det(lattvec))

    if 't' in control:
        temperatures = np.array([control['t']], dtype=float)
    else:
        t_min = control['t_min']
        t_max = control['t_max']
        t_step = control['t_step']
        temperatures = np.arange(t_min, t_max + 1, t_step, dtype=float)

    # discover temperature directories

    temperature_dirs = sorted(
        [p for p in directory.glob('T*K') if p.is_dir()],
        key=lambda d: int(d.name.strip('TK')))

    # determine available quantities

    available = ['temperature', 'qpoint', 'frequency', 'group_velocity',
                 'gruneisen', 'gamma', 'lattice_thermal_conductivity',
                 'lifetime', 'mean_free_path', 'occupation', 'mode_kappa']
    has_3ph = any((td / 'BTE.w_3ph').exists() for td in temperature_dirs)
    has_4ph = any((td / 'BTE.w_4ph').exists() for td in temperature_dirs)
    if has_3ph:
        available.append('gamma_3ph')
    if has_4ph:
        available.append('gamma_4ph')
    available.extend(['cumulative_kappa_mfp', 'cumulative_kappa_frequency'])

    if 'all' in quantities:
        quantities = ['temperature', 'qpoint', 'frequency', 'group_velocity',
                      'gruneisen', 'gamma', 'lattice_thermal_conductivity',
                      'lifetime', 'mean_free_path', 'occupation', 'mode_kappa']

    # initialise data

    data = {'meta': {'kappa_source': 'shengbte',
                     'units':        {},
                     'dimensions':   {}}}

    # read temperature-independent data

    if 'frequency' in quantities or any(
            q in quantities for q in ['mode_kappa', 'occupation']):
        omega = np.loadtxt(directory / 'BTE.omega')
        nq = omega.shape[0]
        data['frequency'] = omega * sconversions['frequency']

    if 'qpoint' in quantities:
        qpoints_file = np.loadtxt(directory / 'BTE.qpoints')
        data['qpoint'] = qpoints_file[:, 3:]
        if 'frequency' not in data:
            nq = data['qpoint'].shape[0]

    if 'group_velocity' in quantities or any(
            q in quantities for q in ['mean_free_path', 'mode_kappa']):
        gv_raw = np.loadtxt(directory / 'BTE.v')
        data['group_velocity'] = gv_raw.reshape(
            (nq, nbands, 3), order='F') * sconversions['group_velocity']

    if 'gruneisen' in quantities:
        grun_path = directory / 'BTE.gruneisen'
        if grun_path.exists():
            data['gruneisen'] = np.loadtxt(grun_path)

    # read temperature-dependent data

    rates_list = []
    rates3_list = []
    rates4_list = []
    kappa_list = []
    ck_mfp_list = []
    ck_freq_list = []

    need_rates = any(q in quantities for q in [
        'gamma', 'lifetime', 'mean_free_path', 'mode_kappa'])
    need_3ph = 'gamma_3ph' in quantities
    need_4ph = 'gamma_4ph' in quantities
    need_kappa = 'lattice_thermal_conductivity' in quantities
    need_ck_mfp = 'cumulative_kappa_mfp' in quantities
    need_ck_freq = 'cumulative_kappa_frequency' in quantities

    raw_rates_list = []

    for td in temperature_dirs:
        if need_rates:
            w_rta = td / 'BTE.w'
            if w_rta.exists():
                rates_raw = np.loadtxt(w_rta, usecols=1)
            else:
                raise FileNotFoundError(
                    'No BTE.w in {}'.format(td))
            rates = rates_raw.reshape((nq, nbands), order='F')
            rates = np.nan_to_num(rates, nan=0.0)
            rates_list.append(rates)

        if need_3ph:
            w3_path = td / 'BTE.w_3ph'
            if w3_path.exists():
                r3 = np.loadtxt(w3_path, usecols=1)
                r3 = r3.reshape((nq, nbands), order='F')
                r3 = np.nan_to_num(r3, nan=0.0)
                rates3_list.append(r3)

        if need_4ph:
            w4_path = td / 'BTE.w_4ph'
            if w4_path.exists():
                r4 = np.loadtxt(w4_path, usecols=1)
                r4 = r4.reshape((nq, nbands), order='F')
                r4 = np.nan_to_num(r4, nan=0.0)
                rates4_list.append(r4)

        if need_kappa:
            kt_file = np.loadtxt(td / 'BTE.kappa_tensor')
            kt = kt_file[-1, 1:].reshape((3, 3))
            kappa_list.append(kt)

        if need_ck_mfp:
            ck_path = td / 'BTE.cumulative_kappa_tensor'
            if ck_path.exists():
                ck_raw = np.loadtxt(ck_path)
                ck_mfp_list.append({
                    'mfp':   ck_raw[:, 0] * 1e-9,
                    'kappa': ck_raw[:, 1:].reshape((-1, 3, 3))})

        if need_ck_freq:
            ckf_path = td / 'BTE.cumulative_kappaVsOmega_tensor'
            if ckf_path.exists():
                ckf_raw = np.loadtxt(ckf_path)
                ck_freq_list.append({
                    'frequency': ckf_raw[:, 0] * sconversions['frequency'],
                    'kappa':     ckf_raw[:, 1:].reshape((-1, 3, 3))})

    # stack temperature-dependent arrays

    if rates_list:
        rates_all = np.array(rates_list)  # (ntemps, nq, nbands) in ps^-1

    if need_kappa and kappa_list:
        data['lattice_thermal_conductivity'] = np.array(kappa_list)

    # compute gamma (HWHM of spectral function) from RTA scattering rates
    # ShengBTE RTA rate w = 1/tau in ps^-1
    # gamma in THz: gamma = w / (4 * pi)
    # This matches phono3py convention: tau = 1 / (4 * pi * gamma)

    if any(q in quantities for q in ['gamma', 'lifetime', 'mean_free_path',
                                      'mode_kappa']):
        data['gamma'] = rates_all / (4.0 * math.pi)

    if need_3ph and rates3_list:
        data['gamma_3ph'] = np.array(rates3_list) / (4.0 * math.pi)

    if need_4ph and rates4_list:
        data['gamma_4ph'] = np.array(rates4_list) / (4.0 * math.pi)

    # compute lifetime: tau = 1/rate, convert ps -> s

    if 'lifetime' in quantities or 'mean_free_path' in quantities or \
       'mode_kappa' in quantities:
        with np.errstate(divide='ignore', invalid='ignore'):
            lifetime = np.reciprocal(rates_all) * 1e-12  # ps -> s
        infmax = np.nanmax(np.where(np.isinf(lifetime), 0, lifetime))
        lifetime = np.where(np.isinf(lifetime), infmax, lifetime)
        lifetime = np.where(np.isnan(lifetime), np.nanmin(lifetime), lifetime)
        if 'lifetime' in quantities:
            data['lifetime'] = lifetime

    # compute mean free path

    if 'mean_free_path' in quantities:
        data['mean_free_path'] = np.multiply(
            lifetime[:, :, :, np.newaxis],
            data['group_velocity'][np.newaxis, :, :, :])

    # compute Bose-Einstein occupation

    if 'occupation' in quantities:
        data['occupation'] = tp.calculate.be_occupation(
            data['frequency'], temperatures, use_tprc=False)

    # compute mode-resolved kappa via RTA kinetic theory
    # kappa_ij(T,q,nu) = (w_q/NV) * C(T,q,nu) * v_i(q,nu) * v_j(q,nu) * tau(T,q,nu)
    # C = kB * x^2 * exp(x) / (exp(x) - 1)^2, x = hbar*omega/(kB*T)
    # w_q = degeneracy of IBZ q-point (number of equivalent points in FBZ)
    # V in m^3 (converted from nm^3 via *1e-27)

    if 'mode_kappa' in quantities:
        hbar = physical_constants['Planck constant over 2 pi'][0]  # J s
        kb = physical_constants['Boltzmann constant'][0]  # J K-1

        freq_hz = data['frequency'] * 1e12 * 2 * math.pi  # THz -> rad/s
        gv = data['group_velocity']  # m/s, (nq, nbands, 3)
        volume_m3 = volume * 1e-27  # nm^3 -> m^3

        # q-point degeneracy from BTE.qpoints (column 2, 0-indexed)
        qpoints_file = np.loadtxt(directory / 'BTE.qpoints')
        degeneracy = qpoints_file[:, 2]  # (nq,)

        mode_kappa_list = []
        for ti, T in enumerate(temperatures):
            if T == 0:
                mk = np.zeros((nq, nbands, 6))
            else:
                x = hbar * freq_hz / (kb * T)  # (nq, nbands)
                with np.errstate(over='ignore', divide='ignore',
                                 invalid='ignore'):
                    exp_x = np.exp(x)
                    C = kb * x**2 * exp_x / (exp_x - 1)**2  # J/K per mode
                C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                tau_t = lifetime[ti]  # (nq, nbands) in s

                # outer product of velocities: v_i * v_j -> 6 Voigt components
                # order: xx, yy, zz, yz, xz, xy
                vv = np.zeros((nq, nbands, 6))
                vv[:, :, 0] = gv[:, :, 0] * gv[:, :, 0]  # xx
                vv[:, :, 1] = gv[:, :, 1] * gv[:, :, 1]  # yy
                vv[:, :, 2] = gv[:, :, 2] * gv[:, :, 2]  # zz
                vv[:, :, 3] = gv[:, :, 1] * gv[:, :, 2]  # yz
                vv[:, :, 4] = gv[:, :, 0] * gv[:, :, 2]  # xz
                vv[:, :, 5] = gv[:, :, 0] * gv[:, :, 1]  # xy

                # prefactor includes degeneracy weight for IBZ -> FBZ
                prefactor = (degeneracy[:, np.newaxis] * C * tau_t
                             / (N * volume_m3))  # (nq, nbands)
                mk = prefactor[:, :, np.newaxis] * vv  # (nq, nbands, 6)
            mode_kappa_list.append(mk)

        data['mode_kappa'] = np.array(mode_kappa_list)  # (ntemps, nq, nbands, 6)

    # store temperature

    if 'temperature' in quantities:
        data['temperature'] = temperatures

    # store cumulative kappa

    if need_ck_mfp and ck_mfp_list:
        data['cumulative_kappa_mfp'] = dict(zip(
            [str(int(t)) for t in temperatures], ck_mfp_list))

    if need_ck_freq and ck_freq_list:
        data['cumulative_kappa_frequency'] = dict(zip(
            [str(int(t)) for t in temperatures], ck_freq_list))

    # convert 6-component Voigt to 3x3 tensors (same as phono3py loader)
    # only applies to mode_kappa which was built in Voigt notation;
    # lattice_thermal_conductivity is already (ntemps, 3, 3) from ShengBTE

    if 'mode_kappa' in data:
        q2 = 'mode_kappa'
        for i, d in enumerate(dims[q2]):
            if d == 6:
                data[q2] = np.moveaxis(data[q2], i, 0)
                data[q2] = [[data[q2][0], data[q2][5], data[q2][4]],
                            [data[q2][5], data[q2][1], data[q2][3]],
                            [data[q2][4], data[q2][3], data[q2][2]]]
                data[q2] = np.moveaxis(data[q2], 0, i+1)
                data[q2] = np.moveaxis(data[q2], 0, i+1)
                dims[q2][i] = 3
                dims[q2].insert(i, 3)
                break

    # lattice_thermal_conductivity is already (ntemps, 3, 3); update dims
    if 'lattice_thermal_conductivity' in data:
        dims['lattice_thermal_conductivity'] = ['temperature', 3, 3]

    # populate meta

    for q2 in data:
        if q2 == 'meta':
            continue
        if q2 in units:
            data['meta']['units'][q2] = units[q2]
        if q2 in dims:
            data['meta']['dimensions'][q2] = dims[q2]

    # add units/dimensions for FourPhonon-specific quantities

    if 'gamma_3ph' in data:
        data['meta']['units']['gamma_3ph'] = 'THz'
        data['meta']['dimensions']['gamma_3ph'] = ['temperature', 'qpoint',
                                                    'band']
    if 'gamma_4ph' in data:
        data['meta']['units']['gamma_4ph'] = 'THz'
        data['meta']['dimensions']['gamma_4ph'] = ['temperature', 'qpoint',
                                                    'band']

    # apply custom conversions from tprc.yaml

    for c in conversions:
        if c in data:
            data[c] = np.multiply(data[c], float(conversions[c]))

    return data

def phonopy_dispersion(filename, xdata=None):
    """Loads phonopy dispersion, and can scale the x values.

    Scaling the x values is necessary to plot multiple dispersions on
    the same axes.

    Arguments
    ---------

        filename : str
            phonopy or sumo band.yaml filepath.

        xdata : dict, optional
            data for the dispersion to scale this to. Should have the
            same path, must have the same number of labels. Not
            necessary if using tp.plot.phonons.add_multi. Default: None.

    Returns
    -------

        dict
            dispersion data.
    """

    import yaml

    pconversions = settings.phonopy_conversions()
    conversions = settings.conversions()

    # load data

    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    x = [d['distance'] for d in data['phonon']]
    qp = [q['q-position'] for q in data['phonon']]
    tickpos, ticks = get_path(data)
    f = [[b['frequency'] for b in p['band']] for p in data['phonon']]

    if xdata is not None:
        # scale data to other path
        x = scale_to_path(x, tickpos, xdata['tick_position'])
        tickpos = xdata['tick_position']

    units = tp.settings.units()
    dimensions = settings.dimensions()
    data2 = {'x':             x,
             'qpoint':        qp,
             'frequency':     f,
             'tick_position': tickpos,
             'tick_label':    ticks,
             'meta':
                 {'phonon_dispersion_source': 'phonopy',
                  'units':      {'frequency': units['frequency']},
                  'dimensions': {'frequency': dimensions['frequency']}}}

    for c in pconversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(pconversions[c]))

    for c in conversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(conversions[c]))

    return data2

def phonopy_dos(filename, poscar='POSCAR', atoms=None):
    """Loads phonopy DoS and collates data per atom and as a total.

    By default reads atom names from a POSCAR, but can be overridden to
    allow for separation of environments.

    Arguments
    ---------

        filename : str
            path to phonopy projected_dos.dat or similar.
        poscar : str, optional
            path to POSCAR. Ignored if atoms specified. Default: POSCAR.
        atoms : str or array-like, optional
            atoms in POSCAR order. Atom names can be repeated, in which
            case their contributions are summed. Numbers can indicate
            repetition in the manner of a chemical formula, so the
            following are all acceptable and equivalent: "Ba 1 Sn 2 O 3",
            "Ba Sn Sn O O O", "Ba Sn O 3". Different environments can be
            distinguised with different atom names.
            Default: read from POSCAR.

    Returns
    -------

        dict
            frequency, DoS per atom and total.
    """

    # load data

    pconversions = settings.phonopy_conversions()
    conversions = settings.conversions()
    units = tp.settings.units()
    dimensions = settings.dimensions()

    data = np.transpose(np.loadtxt(filename))
    n = 0
    d = []
    while np.sum(data[1:,n+1]) == 0.0:
        d.append(n)
        n+=1
    n = -1
    while np.sum(data[1:,n-1]) == 0.0:
        d.append(n)
        n-=1
    data = np.delete(data, d, axis=1)

    data2 = {'frequency': data[0],
             'meta':      {'phonon_dos_source': 'phonopy',
                           'units':      {'frequency': units['frequency']},
                           'dimensions': {'frequency': dimensions['frequency']}}}

    if atoms is None:
        from pymatgen.io.vasp.inputs import Poscar
        poscar = Poscar.from_file(poscar, check_for_POTCAR=False,
                                  read_velocities=False).as_dict()
        atoms = [p['label'] for p in poscar['structure']['sites']]
    elif isinstance(atoms, str):
        atoms = atoms.split()

    # combine atoms contributions

    i = 0
    n = 1
    while i < len(atoms):
        try:
            atoms[i+1] = int(atoms[i+1])
            if atoms[i] in data2:
                data2[atoms[i]] += np.sum(data[n:n+atoms[i+1]], axis=0)
            else:
                data2[atoms[i]] = np.sum(data[n:n+atoms[i+1]], axis=0)
            n += atoms[i+1]
            i += 2
        except Exception:
            if atoms[i] in data2:
                data2[atoms[i]] += data[n]
            else:
                data2[atoms[i]] = data[n]
            n += 1
            i += 1

    data2['total'] = np.sum(data[1:], axis=0)

    for c in pconversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(pconversions[c]))

    for c in conversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(conversions[c]))

    return data2

def phonopy_gruneisen(filename):
    """Loads phonopy gruneisen data.

    Does not load path data, but can load from files with a q-point
    path, which will often be preferable if projecting onto a phonon
    dispersion.

    Arguments
    ---------

        filename : str
            phonopy gruneisen.yaml filepath.

    Returns
    -------

        dict
            gruneisen data.
    """

    import yaml

    pconversions = settings.phonopy_conversions()
    conversions = settings.conversions()

    # load data

    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    
    x = np.reshape([[q['distance'] for q in path['phonon']] for path in data['path']], (-1, 3))
    qp = np.reshape([[q['q-position'] for q in path['phonon']] for path in data['path']], (-1, 3))
    g = [[[band['gruneisen'] for band in q['band']] for q in path['phonon']] for path in data['path']]
    g = np.reshape(g, (-1, np.shape(g)[-1]))

    units = tp.settings.units()
    dimensions = settings.dimensions()
    data2 = {'x':             x,
             'qpoint':        qp,
             'gruneisen':     g,
             'meta':
                 {'gruneisen_source': 'phonopy',
                  'units':      {'gruneisen': units['gruneisen']},
                  'dimensions': {'gruneisen': dimensions['gruneisen']}}}

    for c in pconversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(pconversions[c]))

    for c in conversions:
        if c in data2:
            data2[c] = np.multiply(data2[c], float(conversions[c]))

    return data2

def get_path(yamldata):
    """Extracts the path from a phonopy yaml.

    Works for the old and new phonopy and sumo yaml files.

    Arguments
    ---------

        yamldata : dict
            raw phonopy dispersion data (i.e. from yaml.safe_load).

    Returns
    -------

        list
            x tick ordinates.
        list
            x tick labels.
    """

    tickindex = [1]
    tickindex.extend(np.cumsum(yamldata['segment_nqpoint']))
    tickpos = [yamldata['phonon'][i-1]['distance'] for i in tickindex]

    try: # old phonopy
        ticks = [i[0] for i in yamldata['labels']]
        ticks.append(yamldata['labels'][-1][1])
    except: # new phonopy/ sumo
        ticks = [yamldata['phonon'][i - 1]['label'] for i in tickindex]
    ticks = [r'$\mathregular{\Gamma}$' if i == 'G' or 'gamma' in i.lower() else
             r'$\mathregular{{{}}}$'.format(i.strip('$')) for i in ticks]

    return tickpos, ticks

def scale_to_path(x, tickpos, scalepos):
    """Scales data to a path.

    Useful to make different phonopy runs fit together or to map
    gruneisen data on a phonon dispersion.

    Arguments
    ---------

        x : list
            wavevector ordinates.
        tickpos : list
            tick wavevectors for scaling.
        scalepos : list
            scale tick wavevectors.

    Returns
    -------

        list
            wavevector ordinates.
    """

    n = 0
    # for each x, while within data and between two high-symmetry
    # points, interpolate onto scale data
    for i in range(len(x)):
        while n <= len(tickpos) and \
              not (x[i] >= tickpos[n] and x[i] <= tickpos[n+1]):
            n += 1
        x[i] = scalepos[n] + ((x[i] - tickpos[n]) * (scalepos[n+1] - scalepos[n]) / \
                             (tickpos[n+1] - tickpos[n]))

    return x
