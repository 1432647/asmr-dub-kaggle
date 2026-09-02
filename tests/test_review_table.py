import pytest

from asmrdub.review_table import (
    EDITOR_COLUMNS,
    apply_editor_rows,
    to_editor_rows,
    validate_editor_rows,
)


def _rows():
    return [
        {
            "number": 1,
            "start_time": "00:00:01.000",
            "end_time": "00:00:03.500",
            "duration": 2.5,
            "text": "你好",
            "origin": "こんにちは",
        },
        {
            "number": 2,
            "start_time": "00:00:04.000",
            "end_time": "00:00:05.000",
            "duration": 1.0,
            "text": "再见",
            "origin": "またね",
        },
    ]


def test_editor_rows_have_the_expected_columns():
    rows = to_editor_rows(_rows())
    assert set(rows[0]) == set(EDITOR_COLUMNS)


def test_duration_is_recomputed_not_trusted():
    source = _rows()
    source[0]["duration"] = 999.0  # stale value upstream may leave behind
    assert to_editor_rows(source)[0]["duration"] == pytest.approx(2.5)


def test_timestamps_use_dot_separator():
    rows = to_editor_rows(_rows())
    assert rows[0]["start"] == "00:00:01.000"
    assert rows[0]["end"] == "00:00:03.500"


def test_nan_cells_become_empty_strings():
    source = _rows()
    source[0]["origin"] = float("nan")
    assert to_editor_rows(source)[0]["origin"] == ""


def test_edits_merge_back_by_number():
    merged = apply_editor_rows(
        _rows(), [{"number": 2, "text": "拜拜", "origin": "じゃあね"}]
    )
    assert merged[0]["text"] == "你好"      # untouched
    assert merged[1]["text"] == "拜拜"
    assert merged[1]["origin"] == "じゃあね"


def test_timings_are_never_writable():
    merged = apply_editor_rows(
        _rows(), [{"number": 1, "text": "改了", "start": "99:99:99.999"}]
    )
    assert merged[0]["start_time"] == "00:00:01.000"


def test_unknown_numbers_are_ignored():
    merged = apply_editor_rows(_rows(), [{"number": 42, "text": "幽灵"}])
    assert [row["text"] for row in merged] == ["你好", "再见"]


def test_round_trip_without_changes_is_identity():
    original = _rows()
    merged = apply_editor_rows(original, to_editor_rows(original))
    assert [r["text"] for r in merged] == [r["text"] for r in original]
    assert [r["start_time"] for r in merged] == [r["start_time"] for r in original]


def test_validate_accepts_good_table():
    assert validate_editor_rows(to_editor_rows(_rows())) == []


def test_blank_translation_is_an_error():
    """Upstream would silently emit 100ms of silence and drop the line."""
    problems = validate_editor_rows([{"number": 1, "text": "  "}])
    assert any("为空" in p for p in problems)


def test_duplicate_numbers_flagged():
    problems = validate_editor_rows(
        [{"number": 1, "text": "a"}, {"number": 1, "text": "b"}]
    )
    assert any("重复" in p for p in problems)


def test_empty_table_flagged():
    assert validate_editor_rows([]) == ["复核表为空"]


def test_missing_number_flagged():
    problems = validate_editor_rows([{"text": "没有编号"}])
    assert any("number" in p for p in problems)
