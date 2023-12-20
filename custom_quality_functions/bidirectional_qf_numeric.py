from collections import namedtuple
import pysubgroup as ps
import numpy as np


class BidirectionalQFNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple("BidirectionalQFNumeric_parameters", ("size_sg", "mean", "estimate"))
    mean_tpl = tpl

    @staticmethod
    def bidirectional_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
        return instances_subgroup**a * abs(mean_sg - mean_dataset)

    def __init__(self, a, invert=False, estimator="sum", centroid="mean"):
        self.a = a
        self.invert = invert
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ("size_sg", "mean")
        self.agg = np.mean
        self.tpl = BidirectionalQFNumeric.mean_tpl
        self.read_centroid = lambda x: x.mean
        self.estimator = BidirectionalQFNumeric.SummationEstimator(self)

    def calculate_constant_statistics(self, data, target):
        data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        target_centroid = self.agg(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = self.tpl(data_size, target_centroid, None)
        self.estimator.calculate_constant_statistics(data, target)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return BidirectionalQFNumeric.bidirectional_qf_numeric(
            self.a,
            dataset.size_sg,
            self.read_centroid(dataset),
            statistics.size_sg,
            self.read_centroid(statistics),
        )

    def calculate_statistics(self, subgroup, target, data, statistics=None):  # pylint: disable=unused-argument
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        sg_centroid = 0
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_centroid = self.agg(sg_target_values)
            estimate = self.estimator.get_estimate(
                subgroup, sg_size, sg_centroid, cover_arr, sg_target_values
            )
        else:
            estimate = float("-inf")
        return self.tpl(sg_size, sg_centroid, estimate)

    @staticmethod
    def optimistic_estimate(subgroup, target, data, statistics=None):
        return float("+inf")

    class SummationEstimator:
        r"""\
        This estimator calculates the optimistic estimate as a hypothetical subgroup\
         which contains only instances with value greater than the dataset mean and\
         is of maximal size.
        .. math::
            oe(sg) = \sum_{x \in sg, T(x)>0} (T(sg) - \mu_0)

        From Florian Lemmerich's Dissertation [section 4.2.2.1, Theorem 2 (page 81)]
        """

        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_centroid = None
            self.target_values_greater_centroid = None

        @staticmethod
        def get_data(data, target):  # pylint: disable=unused-argument
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.indices_greater_centroid = (
                self.qf.all_target_values
                > self.qf.read_centroid(self.qf.dataset_statistics)
            )
            self.target_values_greater_centroid = (
                self.qf.all_target_values
            )  # [self.indices_greater_mean]

        def get_estimate(self, subgroup, sg_size, sg_centroid, cover_arr, _):  # pylint: disable=unused-argument
            larger_than_centroid = self.target_values_greater_centroid[cover_arr][
                self.indices_greater_centroid[cover_arr]
            ]
            size_greater_centroid = len(larger_than_centroid)
            sum_greater_centroid = np.sum(larger_than_centroid)

            return sum_greater_centroid - size_greater_centroid * self.qf.read_centroid(
                self.qf.dataset_statistics
            )
