"""Per-robot visitation residue: the spatial half of the graded memory.

The graded memory is ``M = (R, S)``: ``R`` is the bounded record set
(:class:`~swarm_perception.fusion.memory.SetMemory`) and ``S`` this residue —
a monotone set of world-grid cells recording where captures have been seen.
Eviction from ``R`` is demotion, not deletion: a record's spatial footprint
stays in ``S`` forever. ``S`` only ever grows, and merging two residues is a
plain set union — commutative, idempotent, and (unlike the record merge)
associative.

Grid resolution is DERIVED from the capture geometry, never configured:
g = s/2 is the coarsest grid where any unseen gap wide enough for a new
capture contains an unmarked cell.

Marking rule: a cell enters ``S`` iff a crop rect FULLY covers the cell's
nominal ``g x g`` square. Partial overlap marks nothing, so a marked cell is
proof the whole cell was observed. The robot maintains the invariant that
every record it ever incorporated — kept, deduped-away, or later evicted —
had its rect marked here first.

The residue lives on the robot as a plain ``set`` of ``(i, j)`` cells; the
packed bitmap only exists on the wire (``share_visitation``).
"""

from __future__ import annotations

import numpy as np

Rect = tuple[int, int, int, int]


class VisitationResidue:
    """Monotone set of fully-observed grid cells for one robot.

    Cells are ``(i, j)`` column/row indices over a fixed grid covering the
    world: cell ``(i, j)`` spans pixels ``[i*g, (i+1)*g) x [j*g, (j+1)*g)``.
    """

    def __init__(self, world_width: int, world_height: int, coverage_side: float) -> None:
        """Derive the grid from world size and capture square side.

        Args:
            world_width: World width in pixels.
            world_height: World height in pixels.
            coverage_side: Capture square side ``s``; the cell size is
                ``g = s // 2`` (see the module docstring for why).

        Raises:
            ValueError: If the derived cell size is below 1 pixel.
        """
        cell = int(coverage_side // 2)
        if cell < 1:
            raise ValueError(
                f"coverage_side {coverage_side} derives a residue cell below 1px"
            )
        self.cell = cell
        self.grid_width = -(-int(world_width) // cell)  # ceil division
        self.grid_height = -(-int(world_height) // cell)
        self.cells: set[tuple[int, int]] = set()

    @property
    def n_cells(self) -> int:
        """Total grid cells; fixes the wire bitmap length."""
        return self.grid_width * self.grid_height

    @property
    def bitmap_bytes(self) -> int:
        """Packed bitmap size in bytes: ceil(n_cells / 8)."""
        return (self.n_cells + 7) // 8

    def mark_rect(self, rect: Rect) -> None:
        """Mark every cell fully covered by ``rect`` (half-open, pixels).

        A cell ``(i, j)`` is fully covered iff ``i*g >= x1``, ``(i+1)*g <=
        x2`` and the same on the y axis. Rects narrower than one cell mark
        nothing.
        """
        x1, y1, x2, y2 = rect
        g = self.cell
        i_min = max(0, -(-x1 // g))  # ceil(x1 / g)
        j_min = max(0, -(-y1 // g))
        i_max = min(self.grid_width, x2 // g)  # exclusive: last i has (i+1)*g <= x2
        j_max = min(self.grid_height, y2 // g)
        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                self.cells.add((i, j))

    def covered_cells(self, rect: Rect) -> set[tuple[int, int]]:
        """The cells ``mark_rect`` would mark for ``rect`` (for invariants/tests)."""
        probe = VisitationResidue.__new__(VisitationResidue)
        probe.cell = self.cell
        probe.grid_width = self.grid_width
        probe.grid_height = self.grid_height
        probe.cells = set()
        probe.mark_rect(rect)
        return probe.cells

    def to_bitmap(self) -> bytes:
        """Pack the cell set into the canonical wire bitmap.

        Row-major over ``(row j, column i)``, one bit per cell, packed with
        ``np.packbits`` (big-endian bit order), zero-padded to a whole byte.
        """
        flat = np.zeros(self.n_cells, dtype=np.uint8)
        for i, j in self.cells:
            flat[j * self.grid_width + i] = 1
        return np.packbits(flat).tobytes()

    def union_bitmap(self, bitmap: bytes) -> None:
        """Union a peer's wire bitmap into this residue (join-semilattice merge).

        Raises:
            ValueError: If the bitmap length does not match this grid.
        """
        if len(bitmap) != self.bitmap_bytes:
            raise ValueError(
                f"residue bitmap is {len(bitmap)} bytes, expected {self.bitmap_bytes}"
            )
        flat = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8))[: self.n_cells]
        for index in np.flatnonzero(flat):
            j, i = divmod(int(index), self.grid_width)
            self.cells.add((i, j))
