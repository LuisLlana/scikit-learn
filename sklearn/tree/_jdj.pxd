from ..utils._typedefs cimport float64_t, int8_t, intp_t

from ._criterion cimport Criterion

cdef class JDJ(Criterion):
    cdef intp_t[::1] n_classes
    cdef intp_t max_n_classes

    cdef float64_t[:, :, ::1] sum_total    # The sum of the weighted count of each label.
    cdef float64_t[:, :, ::1] sum_left     # Same as above, but for the left side of the split
    cdef float64_t[:, :, ::1] sum_right    # Same as above, but for the right side of the split
    cdef float64_t[:, :, ::1] sum_missing  # Same as above, but for missing values in X
    cdef float64_t[:, ::1] pol_table    # Polarization table
