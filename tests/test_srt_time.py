import pytest

from asmrdub.srt_time import build_srt, format_df, format_srt, format_time, parse_time


@pytest.mark.parametrize(
    "text,expected",
    [
        ("00:00:00,000", 0.0),
        ("00:01:02,500", 62.5),
        ("00:01:02.500", 62.5),
        ("01:00:00,000", 3600.0),
        ("0:00:05", 5.0),
        ("00:00:01,5", 1.5),
        ("00:00:01,050", 1.05),
    ],
)
def test_parse_time_accepts_both_separators(text, expected):
    assert parse_time(text) == pytest.approx(expected)


def test_parse_time_passes_numbers_through():
    assert parse_time(12.25) == 12.25
    assert parse_time(3) == 3.0


def test_parse_time_rejects_garbage():
    for bad in ("", "abc", "1:2", "00:99:00,000", "00:00:61,000"):
        with pytest.raises(ValueError):
            parse_time(bad)


def test_format_round_trips():
    for seconds in (0.0, 1.05, 62.5, 3599.999, 3600.0, 7325.123):
        assert parse_time(format_srt(seconds)) == pytest.approx(seconds, abs=1e-3)


def test_format_separators():
    assert format_srt(62.5) == "00:01:02,500"
    assert format_df(62.5) == "00:01:02.500"


def test_format_clamps_negative():
    assert format_time(-5.0) == "00:00:00,000"


def test_format_truncates_rather_than_rounds():
    """Rounding up could place a cue past the sample it describes."""
    assert format_srt(1.9999) == "00:00:01,999"


def test_build_srt_numbers_from_one():
    srt = build_srt([(0.0, 1.0, "第一句"), (2.0, 3.5, "第二句")])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\n第一句")
    assert "2\n00:00:02,000 --> 00:00:03,500\n第二句" in srt


def test_build_srt_skips_empty_and_renumbers():
    srt = build_srt([(0.0, 1.0, "  "), (2.0, 3.0, "有内容"), (4.0, 5.0, None)])
    assert srt.count("-->") == 1
    assert srt.startswith("1\n")


def test_build_srt_gives_zero_length_cues_a_millisecond():
    srt = build_srt([(5.0, 5.0, "瞬间")])
    assert "00:00:05,000 --> 00:00:05,001" in srt


def test_build_srt_empty_input():
    assert build_srt([]) == ""
