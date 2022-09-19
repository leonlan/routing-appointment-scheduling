from copy import copy


class Solution:
    tour: list[int]
    unassigned: list[int]

    def __init__(self, tour: list[int], unassigned: list[int] = []):
        self.tour = tour
        self.unassigned = unassigned

    def __copy__(self):
        return Solution(copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    def insert(self, job: int, idx: int) -> None:
        """
        Insert the job at position idx.
        """
        self.tour.insert(idx, job)

    def remove(self, job: int) -> None:
        """
        Remove the job from the current schedule.
        """
        self.tour.remove(job)
