from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class LoadStatus(str, Enum):
    OK = "ok"
    NO_OBJECTS = "no_objects"
    ALL_SCORES_NULL = "all_scores_null"
    ACCESS_DENIED = "access_denied"
    READ_ERROR = "read_error"
    SCHEMA_ERROR = "schema_error"


@dataclass
class ArtifactLoadResult:
    status: LoadStatus
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    message: str = ""
    latest_key: str | None = None
    row_count: int = 0
    valid_score_count: int = 0
    latest_dt: datetime | None = None
    source_name: str = ""

    @property
    def ok(self) -> bool:
        return self.status == LoadStatus.OK


@dataclass
class FreshnessLoadResult:
    status: LoadStatus
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == LoadStatus.OK

