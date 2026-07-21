"""Per-robot channel endpoint: broadcast, inbox, and merge-budget fates.

One :class:`ExchangeUnit` per robot owns the communication state — the
bounded FIFO inbox, the delayed-delivery queue, the quantize-once payload
cache, and the per-epoch merge counter — and implements every message fate:

- ``within_budget`` / ``deterministic_after_budget`` — the message merges
  into the robot's record memory (via the canonical
  :func:`~swarm_perception.fusion.merge.peer_merge`).
- ``drop_after_budget`` — the message is discarded over budget, but its
  spatial footprint still enters the visitation residue.
- ``inbox_overflow`` — the FIFO evicts the oldest queued message.
- ``channel_drop`` — the transmission never arrived (Bernoulli drop at the
  sender side).

Every fate logs exactly one priced comm event (byte model:
:mod:`swarm_perception.sim.channel`), so a robot's cumulative spent channel
bytes is the sum over its comm events.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from swarm_perception.fusion.memory import Key
from swarm_perception.fusion.merge import peer_merge
from swarm_perception.perception.encoder import EMBED_DIM
from swarm_perception.sim.channel import Message, build_message, reconstruct_records

if TYPE_CHECKING:
    from swarm_perception.sim.robot import Robot

#: Bounded FIFO inbox depth; overflow evicts the oldest queued message.
INBOX_CAPACITY = 8


class ExchangeUnit:
    """Communication state and message-fate handling for one robot."""

    def __init__(self, robot: Robot) -> None:
        """Bind to the owning robot (config, memory, residue, logger)."""
        self._robot = robot
        self.inbox: deque[Message] = deque()
        self.pending: deque[Message] = deque()  # delayed deliveries, arrival order
        # Quantize-once cache: wire payload per record key. Own records are
        # encoded on first share; received records keep their exact wire bytes.
        self.payload_cache: dict[Key, bytes] = {}
        self.merges_this_epoch = 0

    # ------------------------------------------------------------ broadcast

    def broadcast(self, neighbors: list[Robot]) -> None:
        """Send one budgeted message to every listed peer.

        The message (record selection, quantized payloads, residue bitmap,
        byte cost) is receiver-independent; each transmission draws its own
        packet-drop decision from the channel's dedicated stream. Dropped
        transmissions log their spent bytes with ``dropped: true``.
        """
        robot = self._robot
        comms = robot.cfg.comms
        residue_bitmap = robot.residue.to_bitmap() if comms.share_visitation else None
        message = build_message(
            sender=int(robot.id),
            sender_tick=robot.tick_count,
            epoch=robot.capture_epoch,
            records=robot.memory.records,
            payload_cache=self.payload_cache,
            cfg=comms,
            residue_bitmap=residue_bitmap,
        )
        for neighbor in neighbors:
            if robot.channel.drops_message():
                self._log(
                    message,
                    receiver=int(neighbor.id),
                    receiver_tick=message.delivery_tick,
                    epoch=message.epoch,
                    merged=False,
                    inbox_policy="channel_drop",
                )
                continue
            neighbor.deliver_message(message)

    # ------------------------------------------------------------- receive

    def deliver(self, message: Message) -> None:
        """Accept one transmitted message.

        With ``delay_ticks`` 0 the message enters the FIFO immediately
        (same-tick semantics); otherwise it waits in the pending queue until
        the owning robot's tick reaches ``message.delivery_tick``.
        """
        if self._robot.cfg.comms.delay_ticks == 0:
            self._enqueue(message)
        else:
            self.pending.append(message)

    def collect_due_pending(self) -> None:
        """Move delay-matured messages into the FIFO (arrival order)."""
        while self.pending and self.pending[0].delivery_tick <= self._robot.tick_count:
            self._enqueue(self.pending.popleft())

    def _enqueue(self, message: Message) -> None:
        """Append to the bounded FIFO; overflow evicts (and prices) the oldest."""
        if len(self.inbox) >= INBOX_CAPACITY:
            evicted = self.inbox.popleft()
            self._log(
                evicted,
                receiver=int(self._robot.id),
                receiver_tick=self._robot.tick_count,
                epoch=evicted.epoch,
                merged=False,
                inbox_policy="inbox_overflow",
            )
        self.inbox.append(message)

    # ------------------------------------------------------------- process

    def process_inbox(self) -> None:
        """Resolve at most one queued message per tick.

        Within the per-epoch budget the message merges; over budget,
        ``comms.over_budget`` decides between merging anyway
        (``"deterministic"``) and discarding (``"drop"``). Discarded
        messages still update the residue and log their spent bytes.
        """
        if not self.inbox:
            return
        comms = self._robot.cfg.comms
        if self.merges_this_epoch < comms.max_inbox_merges_per_epoch:
            self._merge(self.inbox.popleft(), inbox_policy="within_budget")
            self.merges_this_epoch += 1
            return
        if comms.over_budget == "deterministic":
            self._merge(self.inbox.popleft(), inbox_policy="deterministic_after_budget")
            return
        message = self.inbox.popleft()
        self._absorb_spatial(message)
        self._log(
            message,
            receiver=int(self._robot.id),
            receiver_tick=self._robot.tick_count,
            epoch=message.epoch,
            merged=False,
            inbox_policy="drop_after_budget",
        )

    def _absorb_spatial(self, message: Message) -> None:
        """Update the residue from a received message BEFORE any merge fate.

        Every record on the wire marks its rect's fully-covered cells, and a
        shared residue bitmap unions in — even when the message itself is
        discarded over budget. The spatial footprint of everything received
        is conserved regardless of what the record merge later evicts.
        """
        residue = self._robot.residue
        for wire_record in message.records:
            residue.mark_rect(wire_record.bbox)
        if message.residue_bitmap is not None:
            residue.union_bitmap(message.residue_bitmap)

    def _merge(self, message: Message, inbox_policy: str) -> None:
        """Merge one message into the record memory and log the comm event."""
        robot = self._robot
        self._absorb_spatial(message)
        for wire_record in message.records:
            # Quantize-once: keep the received wire bytes for re-sharing.
            self.payload_cache.setdefault(wire_record.key, wire_record.payload)
        records = reconstruct_records(message, robot.cfg.comms.quantization, EMBED_DIM)
        robot.memory = peer_merge(robot.memory, records, budget=None)
        self._log(
            message,
            receiver=int(robot.id),
            receiver_tick=robot.tick_count,
            epoch=robot.capture_epoch,
            merged=True,
            inbox_policy=inbox_policy,
        )

    def _log(
        self,
        message: Message,
        *,
        receiver: int,
        receiver_tick: int,
        epoch: int,
        merged: bool,
        inbox_policy: str,
    ) -> None:
        """Emit the single priced comm event for one message fate."""
        self._robot.run_logger.log_comm(
            receiver_tick=receiver_tick,
            sender_tick=message.sender_tick,
            epoch=epoch,
            receiver=receiver,
            sender=message.sender,
            merge_method="deterministic" if merged else "none",
            inbox_policy=inbox_policy,
            bytes_size=message.nbytes,
            k_sent=message.k_sent,
            dropped=not merged,
        )
