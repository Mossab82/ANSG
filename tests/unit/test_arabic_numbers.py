import pytest

from ansg.data.arabic_numbers import parse_arabic_number


@pytest.mark.parametrize("text,expected", [
    ("الأولى", 1), ("الاولى", 1), ("الثانية", 2), ("العاشرة", 10),
    ("الحادية عشرة", 11), ("الثانية عشرة", 12),
    ("الخامسة والعشرون", 25), ("العشرون", 20), ("التاسعة والتسعون", 99),
    ("المائة", 100), ("المئة", 100), ("الأولى بعد المائة", 101),
    ("الحادية والعشرون بعد المائة", 121), ("الثالثة بعد الثلاثمائة", 303),
    ("الثالثة بعد الثلاث مائة", 303), ("العاشرة بعد المائتين", 210),
    ("الأولى بعد الألف", 1001), ("الألف", 1000),
    ("25", 25), ("٢٥", 25), ("۱۲", 12),
    ("الثَّانِيَة", 2),
])
def test_parses(text, expected):
    assert parse_arabic_number(text) == expected


@pytest.mark.parametrize("text", ["", "من الليالي", "الأولى الثانية", "عشرة", "الثلاث", "سعيدة"])
def test_rejects_rather_than_guesses(text):
    assert parse_arabic_number(text) is None
