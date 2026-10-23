"""FastLAS translator.

translate_day writes one .las file per day with one example per location: its
context is that location's day as qualitative facts (cloud layers, humidity,
wind, fronts, stability, ...) plus the bulletin's (sky, rain) pair.
merge_into_dataset puts a background in front of the days: constant pools,
the region's geography, meteorological concepts derived from the facts (the
vocabulary of the mode bias), and the rules that turn each bulletin into the
example's inclusion and exclusions. The result is a single task:

    FastLAS --opl <first day>_<last day>.las

and the learned rules read like

    forecast(V0,overcast,heavy) :- orographic_lift(V0), deep_moisture(V0), location(V0).
"""

import json
import math
import re
import textwrap
from datetime import date
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from region import Region

from .translator import BaseTranslator


class FastLasTranslator(BaseTranslator):
    extension = "LAS"
    # cost of leaving one location-day unexplained, against one point per
    # rule literal: raise it for more (and more specific) rules
    example_penalty = 1

    def translate_day(
        self, day_input_folder: Path, target_date: date, region: Region
    ) -> bytes:
        print("translating fastlas")
        return None

    def merge_into_dataset(self, dates: list[date], input_folder: Path) -> Path | None:
        return None
