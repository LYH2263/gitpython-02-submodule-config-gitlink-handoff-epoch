# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

"""Nested, recoverable handoff generations for submodule state.

GitPython keeps submodule state on three ledgers that historically did not
share a transaction boundary:

* the *configuration* ledger: the ``.gitmodules`` file and the
  ``submodule.<name>`` sections of ``.git/config``;
* the *module* ledger: a submodule's working-tree checkout and its metadata
  directory under ``.git/modules`` (clone, fetch, checkout, deinit);
* the *gitlink* ledger: the mode-``160000`` entry (and the ``.gitmodules``
  blob entry) in the parent repository's index.

This module adds *handoff generations* on top of those ledgers. A generation
is a short-lived scope owned by a host (an embedding application). Mutations
started inside a generation carry an intent stamped with that generation.
Once the host advances to a newer generation, intents of the older generation
can no longer land, whether they target configuration, the module checkout
or the parent gitlink; they fail with :class:`StaleSubmoduleGenerationError`
*before* touching any state. When a generation scope exits through an
exception, the side effects that did land inside it are rolled back.

Generations nest. A nested short-lived generation (for example an inner task
adding or deinitializing another set of submodules while an outer recursive
update is still running) gets its own generation number and its own undo log;
it can neither move the outer cursor nor leave its side effects behind when it
fails. Each :class:`~git.repo.base.Repo` instance owns its own handoff state,
so generations running inside a submodule repository never share a ledger
with generations running in the superproject.

When no generation is active every checkpoint is a no-op, leaving the
behaviour of :class:`Submodule`, :class:`RootModule` and
:meth:`Repo.submodule_update` exactly as it was without this machinery.

The checkpoints deliberately stay on the three native ledgers:
configuration, module checkout and gitlink are fenced independently, and the
differential phase of :meth:`RootModule.update` (remove/path/branch/url
changes between two commits) is fenced separately from the per-submodule
tip-or-pinned checkout command semantics. This module therefore never turns
the ``configuration``/``module`` boolean switches of ``move``/``remove`` into
one consolidated policy switch.
"""

__all__ = [
    "HandoffLedger",
    "HandoffLanding",
    "SubmoduleGenerationError",
    "StaleSubmoduleGenerationError",
    "SubmoduleHandoff",
    "HandoffScope",
    "HandoffIntent",
    "submodule_handoff",
    "submodule_generation",
]

import contextlib
import enum
import logging
import os
import os.path as osp
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, NamedTuple, Optional, Tuple, Union

from git.util import rmtree

if TYPE_CHECKING:
    from git.index import IndexFile
    from git.repo import Repo

_logger = logging.getLogger(__name__)


class HandoffLedger(enum.Enum):
    """The independent ledgers a submodule mutation can land on.

    :ivar CONFIGURATION:
        ``.gitmodules`` and ``submodule.<name>`` sections in ``.git/config``.

    :ivar MODULE:
        The module working-tree checkout and its ``.git/modules`` metadata:
        clone, reconnect, checkout (pinned or tip) and deinit.

    :ivar GITLINK:
        The parent repository index entries for the submodule gitlink and for
        the ``.gitmodules`` blob.

    :ivar ROOT_DIFF:
        The differential phase of :meth:`RootModule.update`, comparing two
        parent commits and applying remove/path/branch/url changes. This phase
        is a different command semantic from the per-submodule
        :meth:`Submodule.update` checkout, and therefore has its own ledger.
    """

    CONFIGURATION = "configuration"
    MODULE = "module"
    GITLINK = "gitlink"
    ROOT_DIFF = "root-diff"


class HandoffLanding(NamedTuple):
    """A record of one mutation that passed a generation fence."""

    generation: int
    ledger: HandoffLedger
    op: str
    name: Optional[str]


class SubmoduleGenerationError(RuntimeError):
    """Base class for failures of the submodule handoff-generation protocol."""


class StaleSubmoduleGenerationError(SubmoduleGenerationError):
    """Raised when an intent of a superseded generation attempts to land.

    The exception is raised before any file, index entry or checkout is
    touched, so the current generation's ledger stays intact. The host may
    open a new generation and retry the operation.
    """


class HandoffScope:
    """One open generation on a :class:`SubmoduleHandoff`.

    Instances are produced by :meth:`SubmoduleHandoff.generation`; they are
    not meant to be constructed directly.
    """

    __slots__ = ("_state", "_number", "_parent", "_undos", "_landings", "_closed")

    def __init__(self, state: "SubmoduleHandoff", number: int, parent: "Optional[HandoffScope]") -> None:
        self._state = state
        self._number = number
        self._parent = parent
        self._undos: list[Callable[[], None]] = []
        self._landings: list[HandoffLanding] = []
        self._closed = False

    #: The generation number of this scope.
    @property
    def number(self) -> int:
        return self._number

    #: The scope that was active when this one was opened, if any.
    @property
    def parent(self) -> "Optional[HandoffScope]":
        return self._parent

    #: ``True`` after the scope has settled or rolled back.
    @property
    def closed(self) -> bool:
        return self._closed

    #: All landings recorded within this scope, in order.
    @property
    def landings(self) -> Tuple[HandoffLanding, ...]:
        return tuple(self._landings)

    def _settle(self) -> None:
        self._closed = True

    def _rollback(self) -> None:
        """Undo every side effect recorded in this scope, last one first."""
        self._closed = True
        undos = self._undos
        self._undos = []
        errors: list[BaseException] = []
        for undo in reversed(undos):
            try:
                undo()
            except Exception as ex:  # noqa: BLE001 - rollback must keep going
                errors.append(ex)
                _logger.error("Submodule handoff rollback action failed", exc_info=True)
        if errors:
            _logger.error(
                "Submodule handoff generation %d rollback completed with %d failed action(s)",
                self._number,
                len(errors),
            )

    def __repr__(self) -> str:
        return "%s(number=%d, closed=%s)" % (type(self).__name__, self._number, self._closed)


class HandoffIntent:
    """A mutation intention stamped with the generation active when it arose.

    An intent is created when a mutating operation begins (a config writer is
    handed out, ``add``/``update``/``move``/``remove``/``deinit`` starts, or
    the root differential phase is entered) and checked at each point where
    the operation actually lands on a ledger. Intents created while no
    generation is active are *legacy* intents whose checkpoints are no-ops,
    which keeps the unmanaged behaviour identical to the unmanaged code.
    """

    __slots__ = ("_state", "_scope", "_ledger", "_frontier", "_name")

    def __init__(
        self,
        state: "SubmoduleHandoff",
        scope: Optional[HandoffScope],
        ledger: HandoffLedger,
        frontier: int,
        name: Optional[str] = None,
    ) -> None:
        self._state = state
        self._scope = scope
        self._ledger = ledger
        self._frontier = frontier
        self._name = name

    #: ``True`` for intents bound to an actual open generation.
    @property
    def enabled(self) -> bool:
        return self._scope is not None

    #: The generation number this intent belongs to, or ``0`` when legacy.
    @property
    def generation(self) -> int:
        return self._scope._number if self._scope is not None else 0

    #: The ledger this intent lands on.
    @property
    def ledger(self) -> HandoffLedger:
        return self._ledger

    def _staleness_reason(self) -> Optional[str]:
        scope = self._scope
        if scope is None:
            return None
        if scope.closed:
            return "closed"
        if self._state.current_scope() is not scope:
            return "superseded"
        if self._state.frontier != self._frontier:
            return "advanced"
        return None

    def check(self, op: str, name: Optional[str] = None) -> bool:
        """Fence a landing point.

        :return:
            ``True`` if the landing belongs to a live generation and was
            recorded; ``False`` for legacy intents (no generation active),
            in which case the caller performs the plain, unmanaged action.

        :raise StaleSubmoduleGenerationError:
            If this intent's generation has settled, been rolled back, been
            covered by a nested newer generation, or been advanced past.
        """
        scope = self._scope
        if scope is None:
            return False
        reason = self._staleness_reason()
        if reason is not None:
            raise StaleSubmoduleGenerationError(
                "submodule handoff generation %d cannot land %s on ledger %s: generation is %s"
                % (scope.number, op, self._ledger.value, reason)
            )
        landing = HandoffLanding(scope.number, self._ledger, op, name if name is not None else self._name)
        scope._landings.append(landing)
        self._state._record_landing(landing)
        return True

    def defer_undo(self, undo: Callable[[], None]) -> None:
        """Register a raw rollback action for this intent's generation.

        No-op for legacy intents. The action must not run through a fenced
        code path: rollback happens after the scope is already closed.
        """
        if self._scope is not None and not self._scope.closed:
            self._scope._undos.append(undo)

    # { Undo primitives

    def remember_index_entry(self, index: "IndexFile", key: Tuple[Any, Any]) -> None:
        """Snapshot an index entry now so it can be restored on rollback."""
        if self._scope is None or self._scope.closed:
            return
        previous = index.entries.get(key)
        repo = self._state.repo

        def undo() -> None:
            fresh = repo.index
            if previous is None:
                fresh.entries.pop(key, None)
            else:
                fresh.entries[key] = previous
            fresh.write()

        self.defer_undo(undo)

    def remember_file(self, path: Any) -> None:
        """Snapshot a file's bytes now so they can be restored on rollback."""
        if self._scope is None or self._scope.closed:
            return
        path = os.fspath(path)
        existed = osp.isfile(path)
        previous = None
        if existed:
            with open(path, "rb") as fp:
                previous = fp.read()

        def undo() -> None:
            if previous is None:
                if osp.isfile(path):
                    os.remove(path)
            else:
                parent = osp.dirname(path)
                if parent and not osp.isdir(parent):
                    os.makedirs(parent, exist_ok=True)
                with open(path, "wb") as fp:
                    fp.write(previous)

        self.defer_undo(undo)

    def remember_creation(self, path: Any, is_dir: bool = True) -> None:
        """Arrange removal of *path* on rollback if it did not exist yet."""
        if self._scope is None or self._scope.closed:
            return
        path = os.fspath(path)
        existed = osp.exists(path) or osp.islink(path)

        def undo() -> None:
            if existed:
                return
            if is_dir:
                if osp.islink(path) or osp.isdir(path):
                    rmtree(path)
            elif osp.isfile(path):
                os.remove(path)

        self.defer_undo(undo)

    # } END undo primitives


class SubmoduleHandoff:
    """Generation registry bound to one (superproject) :class:`~git.repo.base.Repo`.

    Obtain an instance through :func:`submodule_handoff`; the registry is
    shared by all submodules of the same repository instance and is not shared
    with submodule repositories nested inside the checkouts.
    """

    __slots__ = ("_repo", "_frontier", "_top", "_landings")

    def __init__(self, repo: "Repo") -> None:
        self._repo = weakref.ref(repo)
        self._frontier = 0
        self._top: Optional[HandoffScope] = None
        self._landings: list[HandoffLanding] = []

    #: The repository this registry belongs to.
    @property
    def repo(self) -> "Repo":
        repo = self._repo()
        assert repo is not None, "SubmoduleHandoff outlived its repository"
        return repo

    #: Highest generation number ever opened on this registry.
    @property
    def frontier(self) -> int:
        return self._frontier

    #: Number of the currently open (innermost) generation, or ``0``.
    @property
    def current(self) -> int:
        return self._top.number if self._top is not None else 0

    #: Whether any generation scope is currently open.
    @property
    def active(self) -> bool:
        return self._top is not None

    def current_scope(self) -> Optional[HandoffScope]:
        """The innermost open scope, or ``None`` when no generation is active."""
        top = self._top
        return top if top is not None and not top.closed else None

    #: Every landing recorded on this registry, in order, across generations.
    @property
    def landings(self) -> Tuple[HandoffLanding, ...]:
        return tuple(self._landings)

    def _record_landing(self, landing: HandoffLanding) -> None:
        self._landings.append(landing)

    @contextlib.contextmanager
    def generation(
        self, declare: Optional[int] = None, name: Optional[str] = None
    ) -> Iterator[HandoffScope]:
        """Open a new handoff generation.

        :param declare:
            Declare an explicit generation number. It must be strictly greater
            than the registry's current frontier; numbers may be skipped. When
            ``None`` (the default), the frontier is advanced by one.

        :param name:
            Optional descriptive label, currently reserved for debugging.

        On normal exit the generation settles: effects that landed inside it
        remain. On exception the generation rolls back every side effect
        recorded through its intents, in reverse landing order, and the
        exception propagates.
        """
        if declare is None:
            number = self._frontier + 1
        else:
            if not isinstance(declare, int) or isinstance(declare, bool) or declare <= self._frontier:
                raise ValueError(
                    "declared submodule handoff generation %r must be an integer greater than the current frontier %d"
                    % (declare, self._frontier)
                )
            number = declare
        # END handle explicit declaration
        self._frontier = number
        scope = HandoffScope(self, number, self._top)
        self._top = scope
        try:
            yield scope
        except BaseException:
            scope._rollback()
            raise
        else:
            scope._settle()
        finally:
            if self._top is scope:
                self._top = scope._parent

    def claim(self, ledger: HandoffLedger, name: Optional[str] = None) -> HandoffIntent:
        """Stamp an intent for *ledger* at the current generation.

        Returns a legacy no-op intent when no generation is active. The intent
        remembers the generation (and frontier) it was claimed in, so it can
        later detect that the host advanced to a newer generation before the
        operation landed.
        """
        top = self.current_scope()
        if top is None:
            return HandoffIntent(self, None, ledger, self._frontier, name)
        return HandoffIntent(self, top, ledger, self._frontier, name)


_registry: "weakref.WeakKeyDictionary[Repo, SubmoduleHandoff]" = weakref.WeakKeyDictionary()


def submodule_handoff(repo: "Repo") -> SubmoduleHandoff:
    """Return the :class:`SubmoduleHandoff` registry for *repo*.

    One registry exists per repository instance; it is created lazily and
    garbage collected together with the repository.
    """
    handoff = _registry.get(repo)
    if handoff is None:
        handoff = SubmoduleHandoff(repo)
        _registry[repo] = handoff
    return handoff


@contextlib.contextmanager
def submodule_generation(repo: "Repo", declare: Optional[int] = None) -> Iterator[HandoffScope]:
    """Convenience wrapper for ``submodule_handoff(repo).generation(declare)``."""
    with submodule_handoff(repo).generation(declare=declare) as scope:
        yield scope
