"""
Set operations for arrays based on sorting.

Notes
-----

For floating point arrays, inaccurate results may appear due to usual round-off
and floating point comparison issues.

Speed could be gained in some operations by an implementation of
`numpy.sort`, that can provide directly the permutation vectors, thus avoiding
calls to `numpy.argsort`.

Original author: Robert Cimrman

"""
import functools

import numpy as np
from numpy.core import overrides


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'ediff1d', 'intersect1d', 'setxor1d', 'union1d', 'setdiff1d', 'unique',
    'in1d', 'isin'
    ]


def _ediff1d_dispatcher(ary, to_end=None, to_begin=None):
    return (ary, to_end, to_begin)


@array_function_dispatch(_ediff1d_dispatcher)
def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
    ary : array_like
        If necessary, will be flattened before the differences are taken.
    to_end : array_like, optional
        Number(s) to append at the end of the returned differences.
    to_begin : array_like, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : ndarray
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    See Also
    --------
    diff, gradient

    Notes
    -----
    When applied to masked arrays, this function drops the mask information
    if the `to_begin` and/or `to_end` parameters are used.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.ediff1d(x)
    array([ 1,  2,  3, -7])

    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    array([-99,   1,   2, ...,  -7,  88,  99])

    The returned array is always 1D.

    >>> y = [[1, 2, 4], [1, 6, 24]]
    >>> np.ediff1d(y)
    array([ 1,  2, -3,  5, 18])

    """
    # force a 1d array
    ary = np.asanyarray(ary).ravel()

    # enforce that the dtype of `ary` is used for the output
    dtype_req = ary.dtype

    # fast track default case
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]

    if to_begin is None:
        l_begin = 0
    else:
        to_begin = np.asanyarray(to_begin)
        if not np.can_cast(to_begin, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_begin` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_begin = to_begin.ravel()
        l_begin = len(to_begin)

    if to_end is None:
        l_end = 0
    else:
        to_end = np.asanyarray(to_end)
        if not np.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_end = to_end.ravel()
        l_end = len(to_end)

    # do the calculation in place and copy to_begin and to_end
    l_diff = max(len(ary) - 1, 0)
    result = np.empty(l_diff + l_begin + l_end, dtype=ary.dtype)
    result = ary.__array_wrap__(result)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    np.subtract(ary[1:], ary[:-1], result[l_begin:l_begin + l_diff])
    return result


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x


def _unique_dispatcher(ar, return_index=None, return_inverse=None,
                       return_counts=None, axis=None):
    return (ar,)


@array_function_dispatch(_unique_dispatcher)
def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

        .. versionadded:: 1.9.0

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.

        .. versionadded:: 1.13.0

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

        .. versionadded:: 1.9.0

    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    repeat : Repeat elements of an array.

    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    (move the axis to the first dimension to keep the order of the other axes)
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.

    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])

    Return the indices of the original array that give the unique values:

    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'], dtype='<U1')

    Reconstruct the input array from the unique values and inverse:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    Reconstruct the input values from the unique values and counts:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved

    """
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.AxisError(axis, ar.ndim) from None

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype)) from e

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)


def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret


def _intersect1d_dispatcher(
        ar1, ar2, assume_unique=None, return_indices=None):
    return (ar1, ar2)





def _get_bounded(ar1, ar2, return_indices = False, assume_sorted=False):
    """ returns max/min projected ar1 and ar2
    max/min optimization:
    idea: range [123456789]
        ar1 = [123 567  ]
        ar2 = [   45 789]
    We can already discard large portions based on min/max values:
        ar1' =[    567  ]
        ar2' =[    5 7  ]
    
    For sorted the cost is O(log(len(ar1))) + O(log(len(ar2)))
    While it is O(len(ar1))+O(len(ar2)) otherwise
    
    """
    if assume_sorted:
        if ar1[0] < ar2[0]:
            lower1 = np.searchsorted(ar1, ar2[0])
            lower2 = 0
        else:
            lower1 = 0
            lower2 = np.searchsorted(ar2, ar1[0])


        if ar1[-1] < ar2[-1]:
            upper1 = len(ar1)
            upper2 = np.searchsorted(ar2, ar1[-1], side='right')
        else:
            upper1 = np.searchsorted(ar1, ar2[-1], side='right')
            upper2 = len(ar2)
           
           
            
        ar1 = ar1[lower1: upper1]
        ar2 = ar2[lower2: upper2]
        if return_indices:
            return ar1, ar2, slice(lower1, upper1),  slice(lower2, upper2)
        else:
            return ar1, ar2

    else:
        largest = min(np.max(ar1), np.max(ar2))
        lowest = max(np.min(ar1), np.min(ar2))

        mask1 = np.logical_and(ar1 >= lowest, ar1 <= largest)
        mask2 = np.logical_and(ar2 >= lowest, ar2 <= largest)
        ar1 = ar1[mask1]
        ar2 = ar2[mask2]
        if return_indices:
            return ar1, ar2, np.nonzero(mask1)[0], np.nonzero(mask2)[0]
        else:
            return ar1, ar2


def _intersect1d_no_indices(ar1, ar2, assume_unique=False, assume_sorted=False, enable_min_max=False):
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)
    
    # early exit
    if len(ar1)==0 or len(ar2)==0:
        return np.array([], dtype=ar1.dtype) # dtype correct?
    
    # do min/max optimization
    if assume_sorted:
        ar1, ar2 = _get_bounded(ar1, ar2, assume_sorted=assume_sorted)
    
        # early exit, array sizes might have changed
        if len(ar1)==0 or len(ar2)==0:
            return np.array([], dtype=ar1.dtype) # dtype correct?
    
    if assume_unique:
        if assume_sorted:
            ar1 = ar1.ravel()
            ar2 = ar2.ravel()
        else:
            ar1 = ar1.copy()
            ar1.sort()
            ar2 = ar2.copy()
            ar2.sort()

    else:
        if assume_sorted:
            # if the larger array is sorted we can omit dedup in case of using searchsorted
            if len(ar1) <= len(ar2):
                ar1 = _dedup(ar1)
            if len(ar2) <= len(ar1):
                ar2 = _dedup(ar2)
        else:
            # not unique and not sorted
            # in principle sorting the larger array is enough
            # this becomes obsolete if there is a faster dedup
            if len(ar1) <= len(ar2):
                ar1 = np.unique(ar1)
            else:
                ar1 = ar1.copy()
                ar1.sort()
            
            if len(ar2) <= len(ar1):
                ar2 = np.unique(ar2)
            else:
                ar2 = ar2.copy()
                ar2.sort()

    
    #if max(len(ar1), len(ar2)) / min(len(ar1), len(ar2)) > 5.:
        # here we use the variant that is much better for larger array difference
    return _intersect_search(ar1, ar2)

def _intersect_search(ar1, ar2, return_indices=False):
    """ intersect the sorted arrays ar1 and ar2 using searchsorted
        this is achieved by first determining the longer array
        then the smaller array is used to search in the larger array

        it returns the intersection and potentially the indices which lead to the intersection.
    """
    
    # find smaller array
    if len(ar1) < len(ar2):
        smaller_arr = ar1
        longer_arr = ar2
        swapped = False
    else:
        longer_arr = ar1
        smaller_arr = ar2
        swapped = True
    
    # use upper value of longer array to limit search in smaller array
    # can potentially be omitted if it is only called with min_max optimisation first
    i = np.searchsorted(smaller_arr, longer_arr[-1], 'right')
    smaller_arr=smaller_arr[:i]

    # find indices into larger array that lead to sorted arr
    indices_larger = np.searchsorted(longer_arr, smaller_arr, 'left')

    # keep only those that are exactly equal
    mask = smaller_arr==longer_arr[indices_larger]
    indices_larger = indices_larger[mask]

    if return_indices:
        indices_smaller = np.nonzero(mask)[0]
        if swapped:
            return smaller_arr[indices_smaller], indices_larger, indices_smaller
        else:
            return smaller_arr[indices_smaller], indices_smaller, indices_larger
    else:
        return smaller_arr[mask]



    
def _dedup(ar, return_index=False):
    """ keeps only one for each consecutive duplicate element
        code is very similar to np.unique
    """
    ar = np.asanyarray(ar).ravel()
    mask = np.empty(ar.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = ar[1:] != ar[:-1]
    
    if return_index:
        return ar[mask], np.nonzero(mask)[0]
    else:
        return ar[mask]
    
@array_function_dispatch(_intersect1d_dispatcher)
def intersect1d(ar1, ar2, return_indices=False, assume_unique=False, assume_sorted=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
        unique, incorrect results and out-of-bounds indices could result.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.

        .. versionadded:: 1.15.0
    assume_sorted : bool
        If True, the input arrays are both assumed to be sorted, which
        can significantly speed up the calculation. If True but ``ar1`` or ``ar2``
        are not unique, incorrect results and out-of-bounds indices could result.
        Default is False.

        .. versionadded:: 1.20.0

    Returns
    -------
    intersect1d : ndarray
        Sorted 1D array of common and unique elements.
    comm1 : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    comm2 : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.


    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Examples
    --------
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])

    To intersect more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])

    To return the indices of the values common to the input arrays
    along with the intersected values:

    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))

    """

    if return_indices:
        return _intersect1d_with_indices(ar1, ar2, assume_unique, assume_sorted)
    else:
        return _intersect1d_no_indices(ar1, ar2, assume_unique, assume_sorted)
    
def _intersect1d_with_indices(ar1, ar2, assume_unique=False, assume_sorted=False):
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)
    
    ar1 = ar1.ravel()
    ar2 = ar2.ravel()
    
    # early exit
    if len(ar1)==0 or len(ar2)==0:
        return (np.array([], dtype=ar1.dtype), # dtype correct?
                np.array([], dtype=np.uint32), # dtype correct?
                np.array([], dtype=np.uint32)) # dtype correct?
    m1_0 = slice(0, None)
    m2_0 = slice(0, None)
    if assume_sorted:
        # do min/max optimization
        ar1, ar2, m1_0, m2_0 = _get_bounded(ar1, ar2, return_indices=True, assume_sorted=assume_sorted)
        
        # early exit, array sizes might have changed
        if len(ar1)==0 or len(ar2)==0:
            return (np.array([], dtype=ar1.dtype), # dtype correct?
                    np.array([], dtype=np.uint32), # dtype correct?
                    np.array([], dtype=np.uint32)) # dtype correct?
    m1_1 = None
    m2_1 = None
    if assume_unique:
        if not assume_sorted:
            m1_1 = ar1.argsort(kind="mergesort")
            m2_1 = ar2.argsort(kind="mergesort")
            ar1 = ar1[m1_1]
            ar2 = ar2[m2_1]
    else:
        if assume_sorted:
            if len(ar1) <= len(ar2):
                ar1, m1_1 = _dedup(ar1, return_index=True)
            else:
                m1_1 = np.arange(len(ar1))

            if len(ar2) < len(ar1):
                ar2, m2_1 = _dedup(ar2, return_index=True)
            else:
                m2_1 = np.arange(len(ar2))
        else:
            ar1, m1_1 = np.unique(ar1, return_index=True)
            ar2, m2_1 = np.unique(ar2, return_index=True)
    
    int1d, ar1_indices, ar2_indices =  _intersect_search(ar1, ar2, return_indices=True)

    if not m1_1 is None:
        ar1_indices = m1_1[ar1_indices]
        ar2_indices = m2_1[ar2_indices]

    if isinstance(m1_0, slice):
        # arrays were already sorted, only offset values
        if m1_0.start > 0:
            ar1_indices += m1_0.start
        if m2_0.start > 0:
            ar2_indices += m2_0.start
    else:
        ar1_indices = m1_0[ar1_indices]
        ar2_indices = m2_0[ar2_indices]
        
    return int1d, ar1_indices, ar2_indices


def _setxor1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)


@array_function_dispatch(_setxor1d_dispatcher)
def setxor1d(ar1, ar2, assume_unique=False):
    """
    Find the set exclusive-or of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    setxor1d : ndarray
        Sorted 1D array of unique values that are in only one of the input
        arrays.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 2, 4])
    >>> b = np.array([2, 3, 5, 7, 5])
    >>> np.setxor1d(a,b)
    array([1, 4, 5, 7])

    """
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = np.concatenate((ar1, ar2))
    if aux.size == 0:
        return aux

    aux.sort()
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    return aux[flag[1:] & flag[:-1]]


def _in1d_dispatcher(ar1, ar2, assume_unique=None, invert=None):
    return (ar1, ar2)


@array_function_dispatch(_in1d_dispatcher)
def in1d(ar1, ar2, assume_unique=False, invert=False):
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    We recommend using :func:`isin` instead of `in1d` for new code.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.

        .. versionadded:: 1.8.0

    Returns
    -------
    in1d : (M,) ndarray, bool
        The values `ar1[in1d]` are in `ar2`.

    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``np.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to an array, in those cases
    ``asarray(ar2)`` is an object array rather than the expected array of
    contained values.

    .. versionadded:: 1.4.0

    Examples
    --------
    >>> test = np.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = np.in1d(test, states)
    >>> mask
    array([ True, False,  True, False,  True])
    >>> test[mask]
    array([0, 2, 0])
    >>> mask = np.in1d(test, states, invert=True)
    >>> mask
    array([False,  True, False,  True, False])
    >>> test[mask]
    array([1, 5])
    """
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]


def _isin_dispatcher(element, test_elements, assume_unique=None, invert=None):
    return (element, test_elements)


@array_function_dispatch(_isin_dispatcher)
def isin(element, test_elements, assume_unique=False, invert=False):
    """
    Calculates `element in test_elements`, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.

    Parameters
    ----------
    element : array_like
        Input array.
    test_elements : array_like
        The values against which to test each value of `element`.
        This argument is flattened if it is an array or array_like.
        See notes for behavior with non-array-like parameters.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted, as if
        calculating `element not in test_elements`. Default is False.
        ``np.isin(a, b, invert=True)`` is equivalent to (but faster
        than) ``np.invert(np.isin(a, b))``.

    Returns
    -------
    isin : ndarray, bool
        Has the same shape as `element`. The values `element[isin]`
        are in `test_elements`.

    See Also
    --------
    in1d                  : Flattened version of this function.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Notes
    -----

    `isin` is an element-wise function version of the python keyword `in`.
    ``isin(a, b)`` is roughly equivalent to
    ``np.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.

    `element` and `test_elements` are converted to arrays if they are not
    already. If `test_elements` is a set (or other non-sequence collection)
    it will be converted to an object array with one element, rather than an
    array of the values contained in `test_elements`. This is a consequence
    of the `array` constructor's way of handling non-sequence collections.
    Converting the set to a list usually gives the desired behavior.

    .. versionadded:: 1.13.0

    Examples
    --------
    >>> element = 2*np.arange(4).reshape((2, 2))
    >>> element
    array([[0, 2],
           [4, 6]])
    >>> test_elements = [1, 2, 4, 8]
    >>> mask = np.isin(element, test_elements)
    >>> mask
    array([[False,  True],
           [ True, False]])
    >>> element[mask]
    array([2, 4])

    The indices of the matched values can be obtained with `nonzero`:

    >>> np.nonzero(mask)
    (array([0, 1]), array([1, 0]))

    The test can also be inverted:

    >>> mask = np.isin(element, test_elements, invert=True)
    >>> mask
    array([[ True, False],
           [False,  True]])
    >>> element[mask]
    array([0, 6])

    Because of how `array` handles sets, the following does not
    work as expected:

    >>> test_set = {1, 2, 4, 8}
    >>> np.isin(element, test_set)
    array([[False, False],
           [False, False]])

    Casting the set to a list gives the expected result:

    >>> np.isin(element, list(test_set))
    array([[False,  True],
           [ True, False]])
    """
    element = np.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)


def _union1d_dispatcher(ar1, ar2):
    return (ar1, ar2)


@array_function_dispatch(_union1d_dispatcher)
def union1d(ar1, ar2):
    """
    Find the union of two arrays.

    Return the unique, sorted array of values that are in either of the two
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. They are flattened if they are not already 1D.

    Returns
    -------
    union1d : ndarray
        Unique, sorted union of the input arrays.

    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Examples
    --------
    >>> np.union1d([-1, 0, 1], [-2, 0, 2])
    array([-2, -1,  0,  1,  2])

    To find the union of more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([1, 2, 3, 4, 6])
    """
    return unique(np.concatenate((ar1, ar2), axis=None))


def _setdiff1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)


@array_function_dispatch(_setdiff1d_dispatcher)
def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the unique values in `ar1` that are not in `ar2`.

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        Input comparison array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    setdiff1d : ndarray
        1D array of values in `ar1` that are not in `ar2`. The result
        is sorted when `assume_unique=False`, but otherwise only sorted
        if the input is sorted.

    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 2, 4, 1])
    >>> b = np.array([3, 4, 5, 6])
    >>> np.setdiff1d(a, b)
    array([1, 2])

    """
    if assume_unique:
        ar1 = np.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]
