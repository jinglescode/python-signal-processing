import numpy as np


def pick_channels(data: np.ndarray,
                  channel_names: [str],
                  selected_channels: [str],
                  verbose: bool = False) -> np.ndarray:

    picked_ch = pick_channels_mne(channel_names, selected_channels)
    data = data[:,  picked_ch, :]

    if verbose:
        print('picking channels: channel_names',
              len(channel_names), channel_names)
        print('picked_ch', picked_ch)
        print()

    del picked_ch

    return data


def pick_channels_mne(ch_names, include, exclude=[], ordered=False):
    """Pick channels by names.
    Returns the indices of ``ch_names`` in ``include`` but not in ``exclude``.
    Taken from https://github.com/mne-tools/mne-python/blob/master/mne/io/pick.py

    Parameters
    ----------
    ch_names : list of str
        List of channels.
    include : list of str
        List of channels to include (if empty include all available).
        .. note:: This is to be treated as a set. The order of this list
           is not used or maintained in ``sel``.
    exclude : list of str
        List of channels to exclude (if empty do not exclude any channel).
        Defaults to [].
    ordered : bool
        If true (default False), treat ``include`` as an ordered list
        rather than a set, and any channels from ``include`` are missing
        in ``ch_names`` an error will be raised.
        .. versionadded:: 0.18
    Returns
    -------
    sel : array of int
        Indices of good channels.
    See Also
    --------
    pick_channels_regexp, pick_types
    """
    if len(np.unique(ch_names)) != len(ch_names):
        raise RuntimeError('ch_names is not a unique list, picking is unsafe')
    # _check_excludes_includes(include)
    # _check_excludes_includes(exclude)
    if not ordered:
        if not isinstance(include, set):
            include = set(include)
        if not isinstance(exclude, set):
            exclude = set(exclude)
        sel = []
        for k, name in enumerate(ch_names):
            if (len(include) == 0 or name in include) and name not in exclude:
                sel.append(k)
    else:
        if not isinstance(include, list):
            include = list(include)
        if len(include) == 0:
            include = list(ch_names)
        if not isinstance(exclude, list):
            exclude = list(exclude)
        sel, missing = list(), list()
        for name in include:
            if name in ch_names:
                if name not in exclude:
                    sel.append(ch_names.index(name))
            else:
                missing.append(name)
        if len(missing):
            raise ValueError('Missing channels from ch_names required by '
                             'include:\n%s' % (missing,))
    return np.array(sel, int)
