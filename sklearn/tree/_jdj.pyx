"""
Todo: repenesar lo que significand los pesos. No puedo tener el cuadadrado de los pesos.
"""


from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs, INFINITY
from libc.stdio cimport printf, stdout, setbuf
from ._criterion cimport ClassificationCriterion, float64_t, intp_t

import numpy as np
cimport numpy as cnp
cnp.import_array()

from ._criterion cimport Criterion


cdef inline void _move_sums_classification_jdj(
    JDJ criterion,
    float64_t[:, :, ::1] sum_1,
    float64_t[:, :, ::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef intp_t k, c1, c2, n_bytes
    if criterion.n_missing != 0 and put_missing_in_1:
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k]**2 * sizeof(float64_t)
            memcpy(&sum_1[k, 0, 0], &criterion.sum_missing[k, 0, 0], n_bytes)

        for k in range(criterion.n_outputs):
            for c1 in range(criterion.n_classes[k]):
                for c2 in range(criterion.n_classes[k]):
                    sum_2[k, c1, c2] = criterion.sum_total[k, c1, c2] - criterion.sum_missing[k, c1, c2]

        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        # Assigning sum_2 = sum_total for all outputs.
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k]**2 * sizeof(float64_t)
            memset(&sum_1[k, 0, 0], 0, n_bytes)
            memcpy(&sum_2[k, 0, 0], &criterion.sum_total[k, 0, 0], n_bytes)

        weighted_n_1[0] = 0.0
        weighted_n_2[0] = criterion.weighted_n_node_samples


cdef class JDJ(Criterion):

    r"""JDJ Index polarization criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)
        count_[k1, k2] =
    """
    def __cinit__(self, intp_t n_outputs,
                  cnp.ndarray[intp_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=intp_t
            The number of unique classes in each target
        """
        # printf("JDJ __cinit__ v1\n")
        self.start = 0
        self.pos = 0
        self.end = 0
        self.missing_go_to_left = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.n_classes = np.empty(n_outputs, dtype=np.intp)

        cdef intp_t k = 0
        cdef intp_t max_n_classes = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        self.max_n_classes = max_n_classes

        self.pol_table = np.zeros((max_n_classes, max_n_classes), dtype=np.float64)
        cdef cnp.ndarray pol_A= np.linspace(0, 1, max_n_classes, dtype=np.float64)
        cdef cnp.ndarray pol_B= np.linspace(1, 0, max_n_classes, dtype=np.float64)
        for i in range(max_n_classes):
            for j in range(max_n_classes):
                v = max(pol_A[i]*pol_B[j], pol_B[i]*pol_A[j])
                self.pol_table[i, j] = v
                print(f'{self.pol_table[i, j]}', end=' ')
            print()
        #setbuf(stdout, NULL)
        self.sum_total = np.zeros((n_outputs, max_n_classes, max_n_classes),
                                  dtype=np.float64)
        self.sum_left = np.zeros((n_outputs, max_n_classes, max_n_classes),
                                 dtype=np.float64)
        self.sum_right = np.zeros((n_outputs, max_n_classes, max_n_classes),
                                  dtype=np.float64)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs, np.asarray(self.n_classes)), self.__getstate__())


    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end) except -1 nogil:

        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of all samples
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to use in the mask
        end : intp_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef intp_t i, j
        cdef intp_t p, q
        cdef intp_t k
        cdef intp_t c1, c2
        cdef float64_t w = 1.0

        setbuf(stdout, NULL)
        # printf("JDJ init: start: %d end: %d, wieight: %f\n", start, end, self.weighted_n_samples)
        for k in range(self.n_outputs):
            memset(&self.sum_total[k, 0, 0], 0, self.n_classes[k]**2 * sizeof(float64_t))
            # memset(&self.sum_left[k, 0, 0], 0, self.n_classes[k]**2 * sizeof(float64_t))
            # memset(&self.sum_right[k, 0, 0], 0, self.n_classes[k]**2 * sizeof(float64_t))

        for p in range(start, end):
            for q in range(start, end):
                i = sample_indices[p]
                j = sample_indices[q]
                # w is originally set to be 1.0, meaning that if no sample weights
                # are given, the default weight of each sample is 1.0.
                if sample_weight is not None:
                    #
                    # Consultar esto: No sé muy bien lo que será esto, pero
                    # la intuición me dice que el peso de un par es el producto
                    # de los pesos.
                    #
                    w = sample_weight[i] * sample_weight[j]

                # Count weighted class frequency for each target
                for k in range(self.n_outputs):
                    c1 = <intp_t> self.y[i, k]
                    c2 = <intp_t> self.y[j, k]
                    self.sum_total[k, c1, c2] += w
                    # printf("init: %d %d %d %f\n", k, c1, c2, self.sum_total[k, c1, c2])
            # esto de los pesos puede ser complicado.
            if sample_weight is not None:
                w = sample_weight[i]
            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = np.zeros((self.n_outputs, self.max_n_classes, self.max_n_classes),
                                    dtype=np.float64)

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, j, p, q, k, c1, c2
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.sum_missing[0, 0, 0], 0,
               self.max_n_classes**2 * self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            for q in range(self.end - n_missing, self.end):
                i = self.sample_indices[p]
                j = self.sample_indices[q]
                if self.sample_weight is not None:
                    #
                    # Consultar esto de nuevo
                    #
                    w = self.sample_weight[i] * self.sample_weight[j]

                for k in range(self.n_outputs):
                    c1 = <intp_t> self.y[i, k]
                    c2 = <intp_t> self.y[j, k]
                    self.sum_missing[k, c1, c2] += w
            # Esto no lo tengo muy claro
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start
        _move_sums_classification_jdj(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
        )
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end
        _move_sums_classification_jdj(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left
        )
        return 0


    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : intp_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef intp_t pos = self.pos
        cdef intp_t start = self.start
        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing


        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef const float64_t[:] sample_weight = self.sample_weight

        cdef intp_t i, j
        cdef intp_t p, q
        cdef intp_t k
        cdef intp_t c1, c2
        cdef float64_t w = 1.0
        cdef float64_t sr, sl

        # sl = 0.0
        # sr = 0.0
        # for k in range(self.n_outputs):
        #     for c1 in range(self.n_classes[k]):
        #         for c2 in range(self.n_classes[k]):
        #             sl += self.sum_left[k, c1, c2]
        #             sr += self.sum_right[k, c1, c2]
        # printf("JDJ update self.pos: %d seld.end: %d sl: %f sr: %f\n",
        #        self.pos, self.end, sl, sr)

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        #printf("c0: start: %d new_pos: %d pos: %d end: %d %d misssing: %d ", self.start, new_pos, pos, self.end, end_non_missing, self.n_missing)
        # if (new_pos - pos) <= (end_non_missing - new_pos):
        # printf("c1\n")
        for p in range(start, pos):
            for q in range(pos, new_pos):
                i = sample_indices[p]
                j = sample_indices[q]
                if sample_weight is not None:
                    #
                    # Consultar esto de nuevo
                    #
                    w = sample_weight[i] * sample_weight[j]

                for k in range(self.n_outputs):
                    self.sum_left[k, <intp_t> self.y[i, k],
                                  <intp_t> self.y[j, k]] += w
                    self.sum_left[k, <intp_t> self.y[j, k],
                                  <intp_t> self.y[i, k]] += w

        for p in range(new_pos, end_non_missing):
            for q in range(pos, new_pos):
                i = sample_indices[p]
                j = sample_indices[q]
                if sample_weight is not None:
                    #
                    # Consultar esto de nuevo
                    #
                    w = sample_weight[i] * sample_weight[j]

                for k in range(self.n_outputs):
                    self.sum_right[k, <intp_t> self.y[i, k],
                                   <intp_t> self.y[j, k]] -= w
                    self.sum_right[k, <intp_t> self.y[j, k],
                                   <intp_t> self.y[i, k]] -= w

        for p in range(pos, new_pos):
            for q in range(pos, new_pos):
                i = sample_indices[p]
                j = sample_indices[q]
                if sample_weight is not None:
                    #
                    # Consultar esto de nuevo
                    #
                    w = sample_weight[i] * sample_weight[j]

                for k in range(self.n_outputs):
                    self.sum_left[k, <intp_t> self.y[i, k],
                                  <intp_t> self.y[j, k]] += w
                    self.sum_right[k, <intp_t> self.y[i, k],
                                   <intp_t> self.y[j, k]] -= w

        # for k in range(self.n_outputs):
        #     memset(&self.sum_left[k, 0, 0], 0, self.n_classes[k]**2 * sizeof(float64_t))
        #     memset(&self.sum_right[k, 0, 0], 0, self.n_classes[k]**2 * sizeof(float64_t))
        # for p in range(new_pos):
        #     for q in range(new_pos):
        #         i = sample_indices[p]
        #         j = sample_indices[q]
        #         if sample_weight is not None:
        #             #
        #             # Consultar esto de nuevo
        #             #
        #             w = sample_weight[i] * sample_weight[j]

        #         for k in range(self.n_outputs):
        #             self.sum_left[k, <intp_t> self.y[i, k],
        #                           <intp_t> self.y[j, k]] += w


        # for p in range(new_pos, end_non_missing):
        #     for q in range(new_pos, end_non_missing):
        #         i = sample_indices[p]
        #         j = sample_indices[q]
        #         if sample_weight is not None:
        #             #
        #             # Consultar esto de nuevo
        #             #
        #             w = sample_weight[i] * sample_weight[j]
        #         self.sum_right[k, <intp_t> self.y[i, k],
        #                        <intp_t> self.y[j, k]] += w


        for p in range(pos, new_pos):
            i = sample_indices[p]
            if sample_weight is not None:
                w = sample_weight[i]
            self.weighted_n_left += w
            self.weighted_n_right -= w

        # else:
        #     self.reverse_reset()
        #     printf("c2\n")
        #     for p in range(end_non_missing - 1, new_pos - 1, -1):
        #         for q in range(end_non_missing - 1, new_pos - 1, -1):
        #             i = sample_indices[p]
        #             j = sample_indices[q]

        #             if sample_weight is not None:
        #                 #
        #                 # Consultar esto de nuevo
        #                 #
        #                 w = sample_weight[i] * sample_weight[j]

        #             for k in range(self.n_outputs):
        #                 self.sum_left[k, <intp_t> self.y[i, k],
        #                               <intp_t> self.y[j, k]] -= w

        #         if sample_weight is not None:
        #             w = sample_weight[i]
        #         self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        sl = 0.0
        sr = 0.0
        for k in range(self.n_outputs):
            for c1 in range(self.n_classes[k]):
                for c2 in range(self.n_classes[k]):
                    sl += self.sum_left[k, c1, c2]
                    sr += self.sum_right[k, c1, c2]
        #             self.sum_right[k, c1, c2] = self.sum_total[k, c1, c2] - self.sum_left[k, c1, c2]
        #             #printf("update: %d %d %d %f %f\n", k, c1, c2, self.sum_right[k, c1, c2], self.sum_left[k, c1, c2], self.sum_total[k, c1, c2])

        self.pos = new_pos
        # printf("end update %f %f\n", sl, sr)
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the JDJ criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end].
        """
        cdef float64_t jdj, partial
        cdef intp_t k, c1, c2

        jdj = 0.0
        for k in range(self.n_outputs):
            for c1 in range(self.n_classes[k]):
                partial = 0.0
                for c2 in range(self.n_classes[k]):
                    partial += self.sum_total[k, c1, c2] * self.pol_table[c1, c2]
                    #printf("%d %d %f, %f, %f\n",c1, c2, self.sum_total[k, c1, c2], self.pol_table[c1, c2], partial)
                jdj += partial
        ##
        ## Si los pesos son 1, self.weighted_n_node_samples es el número de entradas en la tabla
        jdj /= ((self.weighted_n_node_samples **2) * self.n_outputs)
        # printf('jdj global: %f\n ', jdj)
        return jdj

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address to save the impurity of the left node to
        impurity_right : float64_t pointer
            The memory address to save the impurity of the right node to
        """
        cdef float64_t jdj_left, jdj_right, partial, sum_left, sum_right
        cdef intp_t k, c1, c2

        jdj_left = 0.0
        sum_left = 0.0
        sum_right = 0.0
        for k in range(self.n_outputs):
            for c1 in range(self.n_classes[k]):
                partial = 0.0
                for c2 in range(self.n_classes[k]):
                    sum_left += self.sum_left[k, c1, c2]
                    partial += self.sum_left[k, c1, c2] * self.pol_table[c1, c2]
                    #printf("left: %d %d %f, %f, %f\n",c1, c2, self.sum_left[k, c1, c2], self.pol_table[c1, c2], partial)
                jdj_left += partial
        jdj_left /= ((self.weighted_n_left**2) / self.n_outputs)
        # printf("sum_left: %f jdj_left: %f start: %d pos: %d  end: %d w:%f\n",
        #        sum_left, jdj_left,
        #        self.start, self.pos, self.end,
        #        self.weighted_n_left)


        jdj_right = 0.0
        for k in range(self.n_outputs):
            for c1 in range(self.n_classes[k]):
                partial = 0.0
                for c2 in range(self.n_classes[k]):
                    sum_right += self.sum_right[k, c1, c2]
                    partial += self.sum_right[k, c1, c2] * self.pol_table[c1, c2]
                    # printf("right: %d %d %f, %f, %f\n",c1, c2, self.sum_right[k, c1, c2], self.pol_table[c1, c2], partial)
                jdj_right += partial
        jdj_right /= ((self.weighted_n_right**2) * self.n_outputs)
        # printf("sum_right: %f jdj_right: %f start: %d pos: %d end: %d w:%f\n", sum_right, jdj_right, self.start, self.pos, self.end, self.weighted_n_right)
        # printf("Total: %f %f %f\n", sum_left, sum_right, sum_left + sum_right)

        impurity_left[0] = jdj_left
        impurity_right[0] = jdj_right



    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] and save it into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address which we will save the node value into.
        """
        cdef intp_t k, c1, c2

        for k in range(self.n_outputs):
            for c1 in range(self.n_classes[k]):
                for c2 in range(self.n_classes[k]):
                    #
                    # Consultar esto: no sé como caclular esto
                    dest[c1] += self.sum_total[k, c1, c2]
                dest[c1] /= self.weighted_n_node_samples**2
            #
            # Esto se hace por cada output, es como su hubera una lista de listas
            dest += self.max_n_classes

    cdef inline void clip_node_value(
        self, float64_t* dest, float64_t lower_bound, float64_t upper_bound
    ) noexcept nogil:
        """Clip the values in dest such that predicted probabilities stay between
        `lower_bound` and `upper_bound` when monotonic constraints are enforced.
        Note that monotonicity constraints are only supported for:
        - single-output trees and
        - binary classifications.
        """
        #
        # Consultar esto: No sé muy bien que es lo que se está haciendo
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

        # Values for binary classification must sum to 1.
        dest[1] = 1 - dest[0]

    cdef inline float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Note that monotonicity constraints are only supported for:
        - single-output trees and
        - binary classifications.
        """
        #
        # Consultar esto: No sé muy bien que es lo que se está haciendo
        return (
            (self.sum_left[0, 0, 0] / (2 * self.weighted_n_left**2)) +
            (self.sum_right[0, 0, 0] / (2 * self.weighted_n_right**2))
        )

    cdef inline bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current classification split"""
        cdef:
            float64_t value_left = self.sum_left[0][0][0] / self.weighted_n_left**2
            float64_t value_right = self.sum_right[0][0][0] / self.weighted_n_right**2

        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        cdef float64_t improvement
        self.children_impurity(&impurity_left, &impurity_right)

        #printf("proxy_impurity_improvement %d %d %d %f %f %f %f\n", self.start, self.pos, self.end, self.weighted_n_right, impurity_right, self.weighted_n_left, impurity_left)
        improvement = (- self.weighted_n_right * impurity_right
                       - self.weighted_n_left * impurity_left)
        return improvement


    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float64_t
            The initial impurity of the parent node before the split

        impurity_left : float64_t
            The impurity of the left child

        impurity_right : float64_t
            The impurity of the right child

        Return
        ------
        float64_t : improvement in impurity after the split occurs
        """
        cdef:
            float64_t improvement

        # printf("improvement: %f %f %f %f %f\n", self.weighted_n_node_samples, self.weighted_n_samples, impurity_left, impurity_right, impurity_parent)

        improvement =  ((self.weighted_n_node_samples / self.weighted_n_samples) *
                        (impurity_parent - (self.weighted_n_right /
                                            self.weighted_n_node_samples * impurity_right)
                         - (self.weighted_n_left /
                            self.weighted_n_node_samples * impurity_left)))
        return improvement
