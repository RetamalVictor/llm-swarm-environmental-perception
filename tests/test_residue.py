"""Visitation residue: derived grid, full-cover marking, bitmap round-trip."""

from __future__ import annotations

import pytest

from swarm_perception.sim.residue import VisitationResidue


def test_grid_is_derived_from_coverage_side() -> None:
    res = VisitationResidue(1000, 600, 200.0)
    assert res.cell == 100  # g = s // 2
    assert (res.grid_width, res.grid_height) == (10, 6)
    assert res.n_cells == 60
    assert res.bitmap_bytes == 8  # ceil(60 / 8)


def test_cell_below_one_pixel_rejected() -> None:
    with pytest.raises(ValueError, match="cell"):
        VisitationResidue(100, 100, 1.0)


def test_mark_rect_requires_full_cover() -> None:
    res = VisitationResidue(1000, 1000, 200.0)  # g = 100

    # Aligned 200x200 rect fully covers exactly its four cells.
    res.mark_rect((100, 100, 300, 300))
    assert res.cells == {(1, 1), (2, 1), (1, 2), (2, 2)}

    # Offset rect straddles cells; only the one fully-inside cell marks.
    res.cells.clear()
    res.mark_rect((150, 150, 350, 350))
    assert res.cells == {(2, 2)}

    # A rect narrower than one cell marks nothing.
    res.cells.clear()
    res.mark_rect((10, 10, 90, 90))
    assert res.cells == set()


def test_mark_rect_clips_to_grid() -> None:
    res = VisitationResidue(300, 300, 200.0)  # g = 100, 3x3 grid
    res.mark_rect((-100, -100, 400, 400))
    assert res.cells == {(i, j) for i in range(3) for j in range(3)}


def test_covered_cells_matches_mark_rect_without_mutation() -> None:
    res = VisitationResidue(1000, 1000, 200.0)
    probe = res.covered_cells((100, 100, 300, 300))
    assert probe == {(1, 1), (2, 1), (1, 2), (2, 2)}
    assert res.cells == set()


def test_bitmap_round_trip_and_union() -> None:
    a = VisitationResidue(1000, 1000, 200.0)
    b = VisitationResidue(1000, 1000, 200.0)
    a.mark_rect((0, 0, 200, 200))
    b.mark_rect((800, 800, 1000, 1000))

    bits_a = a.to_bitmap()
    assert len(bits_a) == a.bitmap_bytes

    b.union_bitmap(bits_a)
    assert b.cells == {(0, 0), (1, 0), (0, 1), (1, 1), (8, 8), (9, 8), (8, 9), (9, 9)}

    # Union is idempotent and monotone.
    before = set(b.cells)
    b.union_bitmap(bits_a)
    assert b.cells == before

    # Round trip: bitmap -> cells -> bitmap is stable.
    c = VisitationResidue(1000, 1000, 200.0)
    c.union_bitmap(b.to_bitmap())
    assert c.cells == b.cells
    assert c.to_bitmap() == b.to_bitmap()


def test_union_bitmap_rejects_wrong_length() -> None:
    res = VisitationResidue(1000, 1000, 200.0)
    with pytest.raises(ValueError, match="bitmap"):
        res.union_bitmap(b"\x00")
