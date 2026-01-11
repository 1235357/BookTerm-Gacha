"""
BookTerm Gacha - Base Data Class
=================================

This module provides the base class for all data objects in the application.
It implements serialization and representation methods used across models.

Features:
    - Automatic __repr__ generation for debugging
    - Type-filtered variable extraction (only primitive types)
    - Consistent data serialization pattern

Usage:
    class MyData(BaseData):
        def __init__(self):
            super().__init__()
            self.name = ""
            self.count = 0

Based on KeywordGacha v0.13.1 by neavo
https://github.com/neavo/KeywordGacha
"""

class BaseData():

    _TYPE_FILTER = (int, str, bool, float, list, dict, tuple)

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.get_vars()})"

    def get_vars(self) -> dict:
        return {
            k: v
            for k, v in vars(self).items()
            if isinstance(v, BaseData._TYPE_FILTER)
        }