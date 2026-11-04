# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

__all__ = [
    "sm_section",
    "sm_name",
    "mkhead",
    "find_first_remote_branch",
    "SubmoduleConfigParser",
]

import os
from io import BytesIO
import weakref

import git
from git.config import GitConfigParser
from git.exc import InvalidGitRepositoryError

# typing -----------------------------------------------------------------------

from typing import Any, Sequence, TYPE_CHECKING, Union

from git.types import PathLike

from .handoff import HandoffLedger, StaleSubmoduleGenerationError, submodule_handoff

if TYPE_CHECKING:
    from weakref import ReferenceType

    from git.refs import Head, RemoteReference
    from git.remote import Remote
    from git.repo import Repo

    from .base import Submodule

# { Utilities


def sm_section(name: str) -> str:
    """:return: Section title used in ``.gitmodules`` configuration file"""
    return f'submodule "{name}"'


def sm_name(section: str) -> str:
    """:return: Name of the submodule as parsed from the section name"""
    section = section.strip()
    return section[11:-1]


def mkhead(repo: "Repo", path: PathLike) -> "Head":
    """:return: New branch/head instance"""
    return git.Head(repo, git.Head.to_full_path(path))


def find_first_remote_branch(remotes: Sequence["Remote"], branch_name: str) -> "RemoteReference":
    """Find the remote branch matching the name of the given branch or raise
    :exc:`~git.exc.InvalidGitRepositoryError`."""
    for remote in remotes:
        try:
            return remote.refs[branch_name]
        except IndexError:
            continue
        # END exception handling
    # END for remote
    raise InvalidGitRepositoryError("Didn't find remote branch '%r' in any of the given remotes" % branch_name)


# } END utilities

# { Classes


class SubmoduleConfigParser(GitConfigParser):
    """Catches calls to :meth:`~git.config.GitConfigParser.write`, and updates the
    ``.gitmodules`` blob in the index with the new data, if we have written into a
    stream.

    Otherwise it would add the local file to the index to make it correspond with the
    working tree. Additionally, the cache must be cleared.

    Please note that no mutating method will work in bare mode.

    When the writer is handed out inside a submodule handoff generation, its
    configuration-ledger intent is stamped with that generation. If the host
    advances to a newer generation before the writer is released, the pending
    write and the :meth:`flush_to_index` side effect are rejected with
    :class:`~git.objects.submodule.handoff.StaleSubmoduleGenerationError`
    before they touch the file or the parent index.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._smref: Union["ReferenceType[Submodule]", None] = None
        self._index = None
        self._auto_write = True
        self._handoff_state: Any = None
        self._handoff_intent: Any = None
        super().__init__(*args, **kwargs)

    # { Interface
    def set_submodule(self, submodule: "Submodule") -> None:
        """Set this instance's submodule. It must be called before the first write
        operation begins."""
        self._smref = weakref.ref(submodule)

    def bind_handoff(self, submodule: "Submodule") -> None:
        """Stamp this writer with the handoff generation active on the submodule's
        parent repository, if any.

        Called when a write-enabled parser is handed to a host through
        :meth:`Submodule.config_writer`. The intent snapshots the current
        generation, so a later :meth:`write` can detect that the host advanced
        to a newer generation and refuse to land.
        """
        self.set_submodule(submodule)
        state = submodule_handoff(submodule.repo)
        self._handoff_state = state
        if state.active:
            self._handoff_intent = state.claim(HandoffLedger.CONFIGURATION, name=submodule.name)
        # END bind active generation intent

    def flush_to_index(self) -> None:
        """Flush changes in our configuration file to the index."""
        assert self._smref is not None
        # Should always have a file here.
        assert not isinstance(self._file_or_files, BytesIO)

        sm = self._smref()
        if sm is not None:
            index = self._index
            if index is None:
                index = sm.repo.index
            # END handle index
            index.add([sm.k_modules_file], write=self._auto_write)
            sm._clear_cache()
        # END handle weakref

    # } END interface

    # { Overridden Methods
    def write(self) -> None:  # type: ignore[override]
        if not self._dirty:
            rval = super().write()
            # Pin behaviour flushes even unmutated writers on release, except
            # for writers whose edits were quarantined as stale: those must
            # never touch the parent index again.
            if not getattr(self, "_handoff_quarantined", False):
                self.flush_to_index()
            return rval
        # END handle writers that never mutated

        intent = self._handoff_intent
        state = self._handoff_state
        if intent is not None and not getattr(intent, "enabled", True):
            intent = None
        # END drop legacy markers
        sm = self._smref() if self._smref is not None else None
        if intent is None and state is not None and getattr(state, "active", False) and sm is not None:
            # The writer predates the active generation; land on its ledger.
            intent = state.claim(HandoffLedger.CONFIGURATION, name=sm.name)
        # END adopt a generation opened after parser creation

        if intent is None or not getattr(intent, "enabled", False):
            # Legacy fast path: identical to the unmanaged behaviour.
            rval = super().write()
            self.flush_to_index()
            return rval
        # END legacy fast path

        # Fenced path: the intent belongs to a real generation. Fence before
        # touching either the file or the parent index.
        try:
            intent.check("gitmodules-write", sm.name if sm is not None else None)
        except StaleSubmoduleGenerationError:
            # Quarantine the pending in-memory edits so a later release() or
            # interpreter shutdown cannot make the stale intent land.
            self._dirty = False
            self._handoff_quarantined = True
            raise

        # Snapshot the state we are about to change so scope rollback can put
        # both the file and the index back exactly as they were.
        modules_path: Union[str, None] = None
        if not isinstance(self._file_or_files, BytesIO):
            modules_path = os.fspath(self._file_or_files)
        if modules_path is not None:
            intent.remember_file(modules_path)
        index = self._index
        if index is None and sm is not None:
            index = sm.repo.index
        if index is not None:
            intent.remember_index_entry(index, index.entry_key(sm.k_modules_file, 0))

        rval = super().write()
        # The .gitmodules blob entering the parent index is the flush side
        # effect of this same configuration landing.
        self.flush_to_index()

        def undo_handoff_write() -> None:
            # Re-clear the submodule cache after the file content was restored.
            if sm is not None:
                sm._clear_cache()

        intent.defer_undo(undo_handoff_write)
        return rval

    # END overridden methods


# } END classes
