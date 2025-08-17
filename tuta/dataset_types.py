#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type definitions for the financial table dataset
Matches the exact schema of your dataset
"""

from enum import Enum
from typing import TypedDict, Union


class FiscalYearFlag(str, Enum):
    CURRENT = "C"
    PREVIOUS = "P"
    BEFORE_PREVIOUS = "BP"
    UNKNOWN = "UNK"


class PeriodFlag(str, Enum):
    FQ = "FQ"  # First Quarter
    HY = "HY"  # Half Year
    TQ = "TQ"  # Third Quarter
    FY = "FY"  # Full Year
    UNKNOWN = "UNK"


class QaFlag(str, Enum):
    QUARTER_ONLY = "Q"
    ACCUMULATED = "A"
    UNKNOWN = "UNK"


class DecimalFlag(str, Enum):
    ONE = "0"  # Units (10^0)
    THOUSAND = "-3"  # Thousands (10^3)
    MILLION = "-6"  # Millions (10^6)
    INF = "INF"
    UNKNOWN = "UNK"


class UnitFlag(str, Enum):
    KRW = "KRW"
    USD = "USD"
    JPY = "JPY"
    CNY = "CNY"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    SHARES = "SHARES"
    PERCENT = "PERCENT"
    PURE = "PURE"
    KRWEPS = "KRWEPS"  # KRW Earnings Per Share
    UNKNOWN = "UNK"


class CellTypeFlag(str, Enum):
    """Original cell type flags"""
    HEADER = "header"
    DATA = "data"
    SUM = "sum"
    FORMULA = "formula"
    EMPTY = "empty"
    OTHER = "other"


class CleanedCellTypeFlag(str, Enum):
    """Cleaned/refined cell type flags"""
    HEADER = "header"
    SUM = "sum"
    DATA = "data"
    SECTION = "section"
    SUBSECTION = "subsection"
    DESC = "desc"  # Description
    OTHER = "other"


class ALIGN(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    NONE = "none"


class Style(TypedDict):
    bold: bool
    fill: bool
    align: ALIGN
    border: bool
    indent_px: int
    indent_level: int


class Cell(TypedDict):
    fiscal_year: FiscalYearFlag
    period: PeriodFlag
    qa: QaFlag
    decimal: DecimalFlag
    unit: UnitFlag
    cell_type: Union[CellTypeFlag, CleanedCellTypeFlag]
    value: str
    style: Style


# Export label mappings for model training
LABEL_MAPS = {
    "fiscal_year": [flag.value for flag in FiscalYearFlag],
    "period": [flag.value for flag in PeriodFlag],
    "qa": [flag.value for flag in QaFlag],
    "decimal": [flag.value for flag in DecimalFlag],
    "unit": [flag.value for flag in UnitFlag],
    "cell_type": [flag.value for flag in CleanedCellTypeFlag],
}

# Label descriptions for interpretation
LABEL_DESCRIPTIONS = {
    "fiscal_year": {
        "C": "Current fiscal year",
        "P": "Previous fiscal year",
        "BP": "Before previous fiscal year",
        "UNK": "Unknown fiscal year"
    },
    "period": {
        "FQ": "First quarter",
        "HY": "Half year",
        "TQ": "Third quarter",
        "FY": "Full year",
        "UNK": "Unknown period"
    },
    "qa": {
        "Q": "Quarter only (not accumulated)",
        "A": "Accumulated from beginning of year",
        "UNK": "Unknown accumulation type"
    },
    "decimal": {
        "0": "Units (ones)",
        "-3": "Thousands (×1,000)",
        "-6": "Millions (×1,000,000)",
        "INF": "Infinite/Not applicable",
        "UNK": "Unknown decimal scale"
    },
    "unit": {
        "KRW": "Korean Won",
        "USD": "US Dollar",
        "JPY": "Japanese Yen",
        "CNY": "Chinese Yuan",
        "EUR": "Euro",
        "GBP": "British Pound",
        "CHF": "Swiss Franc",
        "SHARES": "Number of shares",
        "PERCENT": "Percentage",
        "PURE": "Pure number (no unit)",
        "KRWEPS": "KRW Earnings Per Share",
        "UNK": "Unknown unit"
    },
    "cell_type": {
        "header": "Table header cell",
        "sum": "Sum/total cell",
        "data": "Data value cell",
        "section": "Section header",
        "subsection": "Subsection header",
        "desc": "Description cell",
        "other": "Other cell type"
    }
}


def validate_cell(cell_data: dict) -> bool:
    """Validate that a cell dictionary matches the expected schema"""
    required_fields = ["fiscal_year", "period", "qa", "decimal", "unit", "cell_type", "value", "style"]
    
    # Check required fields
    for field in required_fields:
        if field not in cell_data:
            print(f"Missing required field: {field}")
            return False
    
    # Validate enum values
    try:
        if cell_data["fiscal_year"] not in [f.value for f in FiscalYearFlag]:
            print(f"Invalid fiscal_year: {cell_data['fiscal_year']}")
            return False
        if cell_data["period"] not in [f.value for f in PeriodFlag]:
            print(f"Invalid period: {cell_data['period']}")
            return False
        if cell_data["qa"] not in [f.value for f in QaFlag]:
            print(f"Invalid qa: {cell_data['qa']}")
            return False
        if cell_data["decimal"] not in [f.value for f in DecimalFlag]:
            print(f"Invalid decimal: {cell_data['decimal']}")
            return False
        if cell_data["unit"] not in [f.value for f in UnitFlag]:
            print(f"Invalid unit: {cell_data['unit']}")
            return False
        if cell_data["cell_type"] not in [f.value for f in CleanedCellTypeFlag]:
            # Also check original CellTypeFlag for compatibility
            if cell_data["cell_type"] not in [f.value for f in CellTypeFlag]:
                print(f"Invalid cell_type: {cell_data['cell_type']}")
                return False
    except Exception as e:
        print(f"Validation error: {e}")
        return False
    
    return True


def get_label_index(label_type: str, label_value: str) -> int:
    """Get the index of a label value for model training"""
    if label_type in LABEL_MAPS and label_value in LABEL_MAPS[label_type]:
        return LABEL_MAPS[label_type].index(label_value)
    return 0  # Default to first value (usually UNK)


def get_label_from_index(label_type: str, index: int) -> str:
    """Get the label value from an index"""
    if label_type in LABEL_MAPS and 0 <= index < len(LABEL_MAPS[label_type]):
        return LABEL_MAPS[label_type][index]
    return LABEL_MAPS[label_type][0]  # Default to first value (usually UNK)