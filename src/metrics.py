from dataclasses import dataclass, field

import pandas as pd
from intervaltree import IntervalTree


@dataclass
class CNV:
    cnv_type: str
    chr: str
    start: int
    end: int
    childs: list["CNV"] = field(default_factory=list)
    intersected: list["CNV"] = field(default_factory=list)
    incorrect_childs: list["CNV"] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def contains_count(self) -> int:
        return len(self.childs)

    @property
    def contains_percent(self) -> float:
        all_length = sum([cnv.length for cnv in self.childs])
        return all_length / self.length

    @property
    def intersection_count(self) -> int:
        return len(self.intersected)

    @property
    def intersection_percent(self) -> float:
        intersection_length = sum(
            [self.intersection_length(cnv) for cnv in self.intersected]
        )
        return intersection_length / self.length

    @property
    def contains_incorrect_count(self) -> int:
        return len(self.incorrect_childs)

    def intersection_length(self, other_cnv: "CNV") -> int:
        return min(self.end, other_cnv.end) - max(self.start, other_cnv.start) + 1

    def contains(self, other_cnv: "CNV") -> bool:
        return (
            self.start <= other_cnv.start
            and self.end >= other_cnv.end
            and self.chr == other_cnv.chr
            and self.cnv_type == other_cnv.cnv_type
        )

    def contains_incorrect(self, other_cnv: "CNV") -> bool:
        return (
            self.start <= other_cnv.start
            and self.end >= other_cnv.end
            and self.chr == other_cnv.chr
            and self.cnv_type != other_cnv.cnv_type
        )

    def intersects(self, other_cnv: "CNV") -> bool:
        return (
            self.start < other_cnv.end
            and self.end > other_cnv.start
            and self.chr == other_cnv.chr
            and self.cnv_type == other_cnv.cnv_type
            and not self.contains(other_cnv)
        )


class CNVMetric:
    """
    Class for calculating metrics for CNVs.
    Predictions are made for genomic windows of a certain size.
    It's size is usually 50bp so it is relatively small compared to CNVs length.
    This metrics takes all predictions and combine them into bigger predictions (CNVs).
    Then it calculates metrics for these CNVs.
    How many predictions are correct, in 80% correct, how many predictions are incorrect, etc.

    Args:
        df (pd.DataFrame): dataframe with real and predicted CNVs
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_metrics(self):
        real_cnvs = self._extract_real_cnvs()
        predicted_cnvs = self._extract_predicted_cnvs()
        cnvs = self._get_childs_intersecting_and_incorrect(real_cnvs, predicted_cnvs)
        predicted_correctly = self.__predicted_correctly(cnvs)
        predicted_half_correctly = self.__predicted_half_correctly(cnvs)
        intersected_half_correctly = self.__intersected_half_correctly(cnvs)
        predicted_incorrectly = self.__predicted_incorrectly(
            cnvs, intersected_half_correctly
        )
        prediction_cov = self.__prediction_cov(cnvs)
        return {
            "predicted_correctly": predicted_correctly,
            "predicted_half_correctly": predicted_half_correctly,
            "intersected_half_correctly": intersected_half_correctly,
            "predicted_incorrectly": predicted_incorrectly,
            "prediction_cov": prediction_cov,
            "all_true_cnvs": len(
                [cnv for cnv in real_cnvs if cnv.cnv_type != "normal"]
            ),
            "all_predicted_cnvs": len(
                [cnv for cnv in predicted_cnvs if cnv.cnv_type != "normal"]
            ),
        }

    def _extract_real_cnvs(self):
        current_type = self.df["cnv_type"][0]
        start = self.df["start"][0]
        chr = self.df["chr"][0]
        chrs_max_len = 0
        cnvs = []
        # Przechodzenie przez dane wejściowe
        for i in self.df.itertuples():
            if i.cnv_type == current_type and i.chr == chr:
                chrs_max_len = i.end
            elif i.chr != chr:
                end = chrs_max_len
                cnvs.append(CNV(current_type, chr, start, end))
                current_type = i.cnv_type
                start = i.start
                chr = i.chr
                chrs_max_len = 0
            else:
                # Zakończenie bieżącej sekwencji i rozpoczęcie nowej
                end = i.start - 1
                cnvs.append(CNV(current_type, chr, start, end))
                current_type = i.cnv_type
                start = i.start
                chr = i.chr
                end = i.end
                chrs_max_len = i.end
        # Dodanie ostatniego zakresu
        cnvs.append(CNV(current_type, chr, start, chrs_max_len))
        return cnvs

    def _extract_predicted_cnvs(self):
        current_type = self.df["pred"][0]
        start = self.df["start"][0]
        chr = self.df["chr"][0]
        chrs_max_len = 0
        preds = []
        # Przechodzenie przez dane wejściowe
        for i in self.df.itertuples():
            if i.pred == current_type and i.chr == chr:
                chrs_max_len = i.end
            elif i.pred == current_type and i.chr != chr:
                end = chrs_max_len
                preds.append(CNV(current_type, chr, start, end))
                current_type = i.pred
                start = i.start
                chr = i.chr
                chrs_max_len = 0
            else:
                # Zakończenie bieżącej sekwencji i rozpoczęcie nowej
                end = i.start - 1
                preds.append(CNV(current_type, chr, start, end))
                current_type = i.pred
                start = i.start
                chr = i.chr
                end = i.end
                chrs_max_len = i.end
        # Dodanie ostatniego zakresu
        preds.append(CNV(current_type, chr, start, chrs_max_len))
        return preds

    def _get_childs_intersecting_and_incorrect(self, cnvs: list[CNV], preds: list[CNV]):

        # Create an interval tree for CNVs
        cnv_interval_tree = IntervalTree()
        for cnv in cnvs:
            cnv_interval_tree[cnv.start : cnv.end] = cnv

        for pred in preds:
            overlapping_cnvs = cnv_interval_tree[pred.start : pred.end]
            for it_cnv in overlapping_cnvs:
                if it_cnv.data.contains(pred):
                    it_cnv.data.childs.append(pred)
                if it_cnv.data.intersects(pred):
                    it_cnv.data.intersected.append(pred)
                if it_cnv.data.contains_incorrect(pred):
                    it_cnv.data.incorrect_childs.append(pred)

        return cnvs

    def __predicted_correctly(self, cnvs):
        dup = sum(
            [
                1
                for cnv in cnvs
                if cnv.contains_count == 1
                and cnv.contains_percent == 1
                and cnv.cnv_type == "dup"
            ]
        )
        dele = sum(
            [
                1
                for cnv in cnvs
                if cnv.contains_count == 1
                and cnv.contains_percent == 1
                and cnv.cnv_type == "del"
            ]
        )
        return {"dup": dup, "del": dele}

    def __predicted_incorrectly(self, cnvs, intersected_half_correctly):
        predicted_incorrectly = sum(
            [cnv.contains_incorrect_count for cnv in cnvs if cnv.cnv_type == "normal"]
        )
        return predicted_incorrectly

    def __intersected_half_correctly(self, cnvs):
        dup = sum(
            [
                1
                for cnv in cnvs
                if cnv.intersection_percent >= 0.8 and cnv.cnv_type == "dup"
            ]
        )
        dele = sum(
            [
                1
                for cnv in cnvs
                if cnv.intersection_percent >= 0.8 and cnv.cnv_type == "del"
            ]
        )
        return {"dup": dup, "del": dele}

    def __predicted_half_correctly(self, cnvs):
        dup = sum(
            [
                1
                for cnv in cnvs
                if cnv.contains_percent + cnv.intersection_percent >= 0.8
                and cnv.intersection_percent < 0.8
                and cnv.contains_percent != 1
                and cnv.cnv_type == "dup"
            ]
        )
        dele = sum(
            [
                1
                for cnv in cnvs
                if cnv.contains_percent + cnv.intersection_percent >= 0.8
                and cnv.contains_percent != 1
                and cnv.cnv_type == "del"
            ]
        )
        return {"dup": dup, "del": dele}

    def __prediction_cov(self, cnvs):
        return sum(
            [
                cnv.contains_percent + cnv.intersection_percent
                for cnv in cnvs
                if cnv.cnv_type != "normal"
            ]
        ) / len([cnv for cnv in cnvs if cnv.cnv_type != "normal"])
