# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

"""Acceptance tests for nested, recoverable submodule handoff generations.

These tests exercise cross-exit timing and causality on real, fully local
repositories (plain paths and local clones; no remote service is ever
contacted). They deliberately do not reduce the feature to a
``configuration``/``module`` boolean table: every test asserts that the
configuration ledger (``.gitmodules`` and ``.git/config``), the module
checkout ledger (clone/checkout/deinit) and the parent gitlink ledger are
fenced independently, and that :meth:`Submodule.update` tip following is a
different command semantic from the :meth:`RootModule.update` differential
phase.
"""

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from git import Git, Repo, RootModule, Submodule
from git.objects.submodule import (
    HandoffLedger,
    StaleSubmoduleGenerationError,
    submodule_handoff,
)
from git.objects.submodule.base import UpdateProgress


def _commit_file(repo, content, name="file"):
    target = Path(repo.working_tree_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    repo.index.add([str(target.relative_to(repo.working_tree_dir)).replace("\\", "/")])
    return repo.index.commit("Write " + content)


def _wpath(repo, *parts):
    return Path(repo.working_tree_dir, *parts)


def _entry_hexsha(index, path, stage=0):
    return index.entries[index.entry_key(path, stage)].hexsha


@pytest.fixture
def local(tmp_path):
    """A parent repository with one locally cloned submodule pinned at ``cached``."""
    with Repo.init(tmp_path / "source") as source, Repo.init(tmp_path / "parent") as parent:
        source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(source, "initial\n")
        _commit_file(source, "cached\n")
        readme = _wpath(parent, "README")
        readme.write_text("parent\n", encoding="utf-8")
        parent.index.add(["README"])
        parent.index.commit("parent initial")
        submodule = parent.create_submodule(
            "module", "module", source.working_tree_dir, branch="master"
        )
        parent.index.commit("Add submodule")
        with submodule.module() as module:
            yield SimpleNamespace(
                tmp=tmp_path,
                source=source,
                parent=parent,
                submodule=submodule,
                module=module,
            )


def _ops(handoff):
    return [(landing.generation, landing.ledger, landing.op, landing.name) for landing in handoff.landings]


# { Generation lifecycle


def test_repo_submodule_and_root_share_one_handoff(local):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)

    assert parent.submodule_handoff is handoff
    assert RootModule(parent).handoff is handoff
    assert submodule.handoff is handoff
    assert not handoff.active
    assert handoff.current == 0
    assert handoff.frontier == 0
    assert handoff.landings == ()


def test_generation_is_declarable_and_progressive(local):
    handoff = local.parent.submodule_handoff

    with handoff.generation() as first:
        assert first.number == 1
        assert handoff.current == 1
        with handoff.generation(declare=5) as jumped:
            assert jumped.number == 5
            assert handoff.frontier == 5
        assert handoff.current == 1
    assert handoff.frontier == 5
    assert not handoff.active

    with handoff.generation() as next_one:
        assert next_one.number == 6

    with pytest.raises(ValueError, match="greater than the current frontier"):
        with handoff.generation(declare=2):
            pass
    with pytest.raises(ValueError, match="greater than the current frontier"):
        with handoff.generation(declare=6):
            pass


# } END generation lifecycle


# { Required scenario 1:
#   A generation-N .gitmodules/flush_to_index intent must not enter the parent
#   index after advancing to generation N+1.


def test_stale_gitmodules_writer_never_reaches_parent_index(local):
    parent, submodule = local.parent, local.submodule
    worktree = Path(parent.working_tree_dir)
    handoff = submodule_handoff(parent)
    blob_before = _entry_hexsha(parent.index, ".gitmodules")
    gitlink_before = _entry_hexsha(parent.index, "module")

    with handoff.generation():
        writer = submodule.config_writer()

        # Advance while the write intent is pending.
        with handoff.generation():
            with pytest.raises(StaleSubmoduleGenerationError, match="gitmodules-write"):
                with writer as stale:
                    stale.set_value("path", "elsewhere")

        # Neither file nor parent index moved.
        assert "elsewhere" not in (worktree / ".gitmodules").read_text(encoding="utf-8")
        assert _entry_hexsha(parent.index, ".gitmodules") == blob_before
        assert _entry_hexsha(parent.index, "module") == gitlink_before

        # A writer claimed by the outer generation after the advance still
        # lands on its own ledger (the frontier snapshot is fresh), while the
        # stale intent above never did.
        with submodule.config_writer() as fresh:
            fresh.set_value("update", "rebase")

        assert any(
            landing.generation == 1 and landing.ledger is HandoffLedger.CONFIGURATION
            for landing in handoff.landings
        )

    # The stale edit never landed even after both generations settled.
    assert "elsewhere" not in (worktree / ".gitmodules").read_text(encoding="utf-8")
    assert "rebase" in (worktree / ".gitmodules").read_text(encoding="utf-8")
    assert _entry_hexsha(parent.index, "module") == gitlink_before


def test_writer_stays_stale_after_inner_generation_settles(local):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)

    with handoff.generation():
        writer = submodule.config_writer()
        with handoff.generation():
            pass  # Nested generation advances the frontier and settles.

        with pytest.raises(StaleSubmoduleGenerationError, match="advanced"):
            with writer as stale:
                stale.set_value("path", "elsewhere")

        assert "elsewhere" not in (Path(parent.working_tree_dir) / ".gitmodules").read_text(encoding="utf-8")


def test_submodule_generation_classmethod_matches_property(local):
    parent, submodule = local.parent, local.submodule
    handoff = parent.submodule_handoff

    with Submodule.handoff_generation(parent) as first:
        assert first.number == 1
        writer = submodule.config_writer()
        with parent.submodule_generation() as nested:
            assert nested.number == 2
            with pytest.raises(StaleSubmoduleGenerationError):
                with writer as stale:
                    stale.set_value("path", "elsewhere")

        # The outer cursor is still usable after the inner scope settles.
        submodule.deinit(force=True)
        assert (
            1,
            HandoffLedger.MODULE,
            "deinit",
            "module",
        ) in _ops(handoff)
        assert "elsewhere" not in (
            Path(parent.working_tree_dir) / ".gitmodules"
        ).read_text(encoding="utf-8")


# } END required scenario 1


# { Required scenario 2:
#   Within one generation, module checkouts fork from configuration and
#   gitlink rewrites.


def test_pinned_checkout_update_lands_module_ledger_only(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    module.head.reset("HEAD~1", index=True, working_tree=True)
    blob_before = _entry_hexsha(parent.index, ".gitmodules")
    gitlink_before = _entry_hexsha(parent.index, "module")
    handoff = submodule_handoff(parent)

    with handoff.generation():
        submodule.update(no_fetch=True)

    ops = _ops(handoff)
    assert (1, HandoffLedger.MODULE, "checkout", "module") in ops
    assert not any(landing.ledger is HandoffLedger.CONFIGURATION for landing in handoff.landings)
    assert not any(landing.ledger is HandoffLedger.GITLINK for landing in handoff.landings)
    assert not any(landing.ledger is HandoffLedger.ROOT_DIFF for landing in handoff.landings)

    # The checkout moved, the parent index did not.
    assert module.head.commit.binsha == submodule.binsha
    assert _entry_hexsha(parent.index, ".gitmodules") == blob_before
    assert _entry_hexsha(parent.index, "module") == gitlink_before


def test_tip_following_is_module_ledger_and_never_a_gitlink_rewrite(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    remote_tip = _commit_file(local.source, "remote-only\n")
    handoff = submodule_handoff(parent)

    with handoff.generation():
        submodule.update(to_latest_revision=True)

    ops = _ops(handoff)
    assert (1, HandoffLedger.MODULE, "tip", "module") in ops
    assert not any(landing.op == "checkout" for landing in handoff.landings)
    assert not any(landing.ledger is HandoffLedger.ROOT_DIFF for landing in handoff.landings)
    assert module.head.commit == remote_tip
    # Following the tip deliberately does not rewrite the parent gitlink.
    assert _entry_hexsha(parent.index, "module") != remote_tip.hexsha


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(module=True, configuration=False), id="module-only"),
        pytest.param(dict(module=False, configuration=True), id="configuration-only"),
    ],
)
def test_move_ledgers_fork(local, kwargs):
    parent, submodule = local.parent, local.submodule
    # Name differs from path, so move() cannot auto-rename the submodule.
    with Repo.init(local.tmp / "moved-src") as moved_source:
        moved_source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(moved_source, "moved source\n")
        parent2 = Repo.init(local.tmp / "moved-parent")
        readme = Path(parent2.working_tree_dir) / "README"
        readme.write_text("x\n", encoding="utf-8")
        parent2.index.add(["README"])
        parent2.index.commit("init")
        sm = parent2.create_submodule(
            "mymod", "subs/mymod", moved_source.working_tree_dir, branch="master"
        )
        parent2.index.commit("add")

        with sm.module():
            handoff = submodule_handoff(parent2)
            with handoff.generation():
                sm.move("relocated", **kwargs)

        ledgers = {landing.ledger for landing in handoff.landings}
        if kwargs["module"]:
            assert HandoffLedger.MODULE in ledgers
            assert HandoffLedger.CONFIGURATION not in ledgers
            assert HandoffLedger.GITLINK not in ledgers
            assert (Path(parent2.working_tree_dir) / "relocated").is_dir()
            assert not (Path(parent2.working_tree_dir) / "subs" / "mymod").exists()
        else:
            assert HandoffLedger.MODULE not in ledgers
            assert HandoffLedger.GITLINK in ledgers
            assert HandoffLedger.CONFIGURATION in ledgers
            assert (Path(parent2.working_tree_dir) / "subs" / "mymod").is_dir()
            assert not (Path(parent2.working_tree_dir) / "relocated").exists()
        parent2.close()


def test_stale_tip_checkout_does_not_move_module(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    _commit_file(local.source, "remote-only\n")
    pinned = module.head.commit.hexsha
    handoff = submodule_handoff(parent)
    stack = contextlib.ExitStack()

    class Advancer(UpdateProgress):
        advanced = False

        def update(self, op_code, cur_count, max_count=None, message=""):
            if not Advancer.advanced and op_code & UpdateProgress.UPDWKTREE and op_code & UpdateProgress.BEGIN:
                stack.enter_context(handoff.generation())
                Advancer.advanced = True

    with handoff.generation():
        try:
            with pytest.raises(StaleSubmoduleGenerationError, match="superseded"):
                submodule.update(to_latest_revision=True, progress=Advancer())
        finally:
            stack.close()

    assert Advancer.advanced
    assert module.head.commit.hexsha == pinned
    assert not any(
        landing.ledger is HandoffLedger.MODULE and landing.op == "tip"
        for landing in handoff.landings
    )


# } END required scenario 2


# { Required scenario 3:
#   Submodule.update tip following forks from RootModule path/url differential
#   changes; one can never gate the other.


def test_root_url_change_records_root_diff_distinct_from_module_checkout(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    previous = parent.head.commit
    module.head.reset("HEAD~1", index=True, working_tree=True)

    with local.source.clone(local.tmp / "mirror") as mirror:
        mirror_url = Git.polish_url(str(mirror.working_tree_dir))
        with submodule.config_writer() as writer:
            writer.set_value("url", mirror_url)
        parent.index.commit("Change submodule URL")

        handoff = submodule_handoff(parent)
        with handoff.generation():
            RootModule(parent).update(previous_commit=previous, recursive=False, no_fetch=True)

    ops = _ops(handoff)
    assert (1, HandoffLedger.ROOT_DIFF, "differential", None) in ops
    assert (1, HandoffLedger.ROOT_DIFF, "url", "module") in ops
    assert (1, HandoffLedger.MODULE, "checkout", "module") in ops
    assert not any(op == "tip" for _, _, op, _ in ops)
    assert not any(landing.ledger is HandoffLedger.GITLINK for landing in handoff.landings)

    assert module.remotes.origin.url == mirror_url
    assert module.head.commit.binsha == submodule.binsha


def test_root_branch_change_is_its_own_root_diff_op(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    branch_name = "master-other"
    local.source.create_head(branch_name)
    module.remotes.origin.fetch()
    previous = parent.head.commit
    with submodule.config_writer() as writer:
        writer.set_value("branch", branch_name)
    parent.index.commit("Change submodule branch")
    module.head.reset("HEAD~1", index=True, working_tree=True)

    handoff = submodule_handoff(parent)
    with handoff.generation():
        RootModule(parent).update(previous_commit=previous, recursive=False, no_fetch=True)

    ops = _ops(handoff)
    assert (1, HandoffLedger.ROOT_DIFF, "branch", "module") in ops
    assert not any(op == "url" for _, _, op, _ in ops)
    assert module.head.reference.name == branch_name


def test_stale_trailing_module_update_does_not_undo_root_url_diff(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    previous = parent.head.commit
    module.head.reset("HEAD~1", index=True, working_tree=True)

    with local.source.clone(local.tmp / "mirror2") as mirror:
        mirror_url = Git.polish_url(str(mirror.working_tree_dir))
        with submodule.config_writer() as writer:
            writer.set_value("url", mirror_url)
        parent.index.commit("Change submodule URL")

        handoff = submodule_handoff(parent)
        stack = contextlib.ExitStack()

        class Advancer(UpdateProgress):
            advanced = False

            def update(self, op_code, cur_count, max_count=None, message=""):
                if (
                    not Advancer.advanced
                    and op_code & UpdateProgress.UPDWKTREE
                    and op_code & UpdateProgress.BEGIN
                ):
                    stack.enter_context(handoff.generation())
                    Advancer.advanced = True

        with handoff.generation():
            try:
                with pytest.raises(StaleSubmoduleGenerationError):
                    RootModule(parent).update(
                        previous_commit=previous, recursive=False, progress=Advancer()
                    )
            finally:
                stack.close()

    # The ROOT_DIFF url phase landed before the trailing MODULE checkout went
    # stale and is not rolled back or gated by the module-ledger failure.
    assert module.remotes.origin.url == mirror_url
    diff_ops = [landing.op for landing in handoff.landings if landing.ledger is HandoffLedger.ROOT_DIFF]
    assert "differential" in diff_ops and "url" in diff_ops
    assert not any(
        landing.ledger is HandoffLedger.MODULE and landing.op in ("checkout", "tip")
        for landing in handoff.landings
    )
    # The stale trailing checkout did not move the module.
    assert module.head.commit.binsha != submodule.binsha


# } END required scenario 3


# { Required scenario 4:
#   Inner short-lived generations must not contaminate outer cursors; an inner
#   failure bounces clean without touching outer state.


def test_inner_generation_failure_bounces_clean_and_keeps_outer_cursor(local, tmp_path):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)

    with Repo.init(tmp_path / "inner-source") as inner_source:
        inner_source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(inner_source, "inner\n")

        with handoff.generation() as outer:
            assert outer.number == 1

            with pytest.raises(RuntimeError, match="inner task failure"):
                with handoff.generation() as inner:
                    assert inner.number == 2
                    inner_sm = parent.create_submodule(
                        "other", "other", inner_source.working_tree_dir, branch="master"
                    )
                    assert parent.index.entry_key("other", 0) in parent.index.entries
                    raise RuntimeError("inner task failure")

            # The inner generation cleaned all three of its own ledgers.
            assert not (Path(parent.working_tree_dir) / "other").exists()
            assert not Path(parent.git_dir, "modules", "other").exists()
            assert parent.index.entry_key("other", 0) not in parent.index.entries
            assert "other" not in (Path(parent.working_tree_dir) / ".gitmodules").read_text(encoding="utf-8")
            assert not parent.config_reader().has_section('submodule "other"')

            # The outer cursor survived; the settled inner still advanced the
            # frontier, so an outer intent from before the inner is stale.
            assert handoff.current == 1
            assert all(landing.generation == 1 for landing in outer.landings)

            # A fresh outer operation lands normally on generation 1.
            submodule.deinit(force=True)
            assert (
                1,
                HandoffLedger.MODULE,
                "deinit",
                "module",
            ) in _ops(handoff)

    # deinit retains metadata and .gitmodules, unlike remove(configuration=True).
    assert Path(parent.git_dir, "modules", "module").is_dir()
    assert (Path(parent.working_tree_dir) / ".gitmodules").is_file()


def test_inner_gitmodules_writer_cannot_flush_into_outer_ledger(local, tmp_path):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)
    blob_before = _entry_hexsha(parent.index, ".gitmodules")

    with handoff.generation():
        with handoff.generation() as inner:
            writer = submodule.config_writer()
            assert inner.number == 2
        # Inner generation settled while its writer intent is pending.

        with pytest.raises(StaleSubmoduleGenerationError):
            with writer as stale:
                stale.set_value("path", "inner-escape")

    assert "inner-escape" not in (Path(parent.working_tree_dir) / ".gitmodules").read_text(encoding="utf-8")
    assert _entry_hexsha(parent.index, ".gitmodules") == blob_before


def test_generations_of_module_repo_never_mix_with_superproject(local, tmp_path):
    parent, submodule, module = local.parent, local.submodule, local.module
    parent_handoff = submodule_handoff(parent)
    module_handoff = submodule_handoff(module)

    with Repo.init(tmp_path / "kid-source") as kid_source:
        kid_source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(kid_source, "kid\n")

        with parent_handoff.generation() as parent_scope:
            with module_handoff.generation() as module_scope:
                assert parent_scope.number == 1 and module_scope.number == 1
                child = module.create_submodule(
                    "kid", "kid", kid_source.working_tree_dir, branch="master"
                )
                assert (Path(module.working_tree_dir) / "kid").is_dir()

        # The inner repository's landings never leaked into the superproject.
        assert parent_handoff.landings == ()
        names = {landing.name for landing in module_handoff.landings}
        assert names == {"kid"}
        assert any(landing.ledger is HandoffLedger.MODULE for landing in module_handoff.landings)
        assert any(landing.ledger is HandoffLedger.GITLINK for landing in module_handoff.landings)
        # A normal settle keeps the inner submodule in place.
        assert (Path(module.working_tree_dir) / "kid").is_dir()
        assert module.index.entry_key("kid", 0) in module.index.entries


def test_inner_module_repo_rollback_does_not_touch_superproject(local, tmp_path):
    parent, submodule, module = local.parent, local.submodule, local.module
    parent_handoff = submodule_handoff(parent)
    module_handoff = submodule_handoff(module)

    with Repo.init(tmp_path / "kid-source2") as kid_source:
        kid_source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(kid_source, "kid\n")

        with parent_handoff.generation():
            with pytest.raises(RuntimeError, match="inner recursive task"):
                with module_handoff.generation():
                    module.create_submodule(
                        "kid2", "kid2", kid_source.working_tree_dir, branch="master"
                    )
                    raise RuntimeError("inner recursive task")

            # The inner rollback is scoped to the module repository.
            assert not (Path(module.working_tree_dir) / "kid2").exists()
            assert module.index.entry_key("kid2", 0) not in module.index.entries
            # The superproject is exactly as the outer generation left it.
            assert parent_handoff.landings == ()
            assert parent.index.entry_key("module", 0) in parent.index.entries
            assert (Path(parent.working_tree_dir) / ".gitmodules").is_file()


def test_recursive_root_update_keeps_nested_repo_generation_apart(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    local.module.git.symbolic_ref("HEAD", "refs/heads/master")
    child = module.create_submodule(
        "nested", "nested", local.source.working_tree_dir, branch="master"
    )
    with child.module() as nested:
        cached_tip = nested.head.commit
        nested.head.reset("HEAD~1", index=True, working_tree=True)
        child.binsha = nested.head.commit.binsha
        module.index.add([child])
        previous = module.index.commit("Add nested at initial commit")
        child.binsha = cached_tip.binsha
        module.index.add([child])
        target = module.index.commit("Advance nested submodule")
        submodule.binsha = target.binsha
        parent.index.add([submodule])
        parent.index.commit("Record nested submodule update")
        module.head.reset(previous, index=True, working_tree=True)

        handoff = submodule_handoff(parent)
        with handoff.generation():
            RootModule(parent).update(
                previous_commit=parent.head.commit.parents[0], recursive=True, no_fetch=True
            )

    parent_names = {landing.name for landing in handoff.landings}
    assert "module" in parent_names
    assert "nested" not in parent_names
    assert module.head.commit == target
    with child.module() as nested_after:
        assert nested_after.head.commit == cached_tip


# } END required scenario 4


# { Rollback and recovery


def test_failed_add_rolls_back_all_three_ledgers(local, tmp_path):
    parent = local.parent
    handoff = submodule_handoff(parent)

    with Repo.init(tmp_path / "add-source") as add_source:
        add_source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(add_source, "add\n")

        with pytest.raises(RuntimeError, match="host aborted after add"):
            with handoff.generation():
                created = parent.create_submodule(
                    "added", "added", add_source.working_tree_dir, branch="master"
                )
                assert (Path(parent.working_tree_dir) / "added").is_dir()
                assert parent.index.entry_key("added", 0) in parent.index.entries
                raise RuntimeError("host aborted after add")

    assert not (Path(parent.working_tree_dir) / "added").exists()
    assert not Path(parent.git_dir, "modules", "added").exists()
    assert parent.index.entry_key("added", 0) not in parent.index.entries
    assert not (Path(parent.working_tree_dir) / ".gitmodules").exists() or "added" not in (
        Path(parent.working_tree_dir) / ".gitmodules"
    ).read_text(encoding="utf-8")
    assert not parent.config_reader().has_section('submodule "added"')


def test_failed_deinit_reinitializes_checkout(local):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)

    with pytest.raises(RuntimeError, match="after deinit"):
        with handoff.generation():
            submodule.deinit(force=True)
            assert not Path(submodule.abspath, ".git").exists()
            raise RuntimeError("after deinit")

    assert Path(submodule.abspath, ".git").is_file()
    assert Path(submodule.abspath, "file").read_text(encoding="utf-8") == "cached\n"


def test_failed_configuration_remove_restores_gitlink(local):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)
    gitlink_before = _entry_hexsha(parent.index, "module")
    modules_text = (Path(parent.working_tree_dir) / ".gitmodules").read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="after remove"):
        with handoff.generation():
            submodule.remove(module=False, configuration=True)
            assert parent.index.entry_key("module", 0) not in parent.index.entries
            raise RuntimeError("after remove")

    assert _entry_hexsha(parent.index, "module") == gitlink_before
    assert (
        Path(parent.working_tree_dir) / ".gitmodules"
    ).read_text(encoding="utf-8") == modules_text
    assert (Path(parent.working_tree_dir) / "module").is_dir()


def test_stale_gitlink_remove_is_blocked(local):
    parent, submodule = local.parent, local.submodule
    handoff = submodule_handoff(parent)
    gitlink_before = _entry_hexsha(parent.index, "module")
    import git.objects.submodule.base as submodule_base

    def advance_without_deleting(path, *args, **kwargs):
        # Stand in for the irreversible module-ledger tree deletion: advance
        # the handoff while remove() is between its MODULE and its
        # CONFIGURATION/GITLINK phases, but keep the checkout on disk so the
        # assertion is independent of platform file-locking semantics.
        with handoff.generation():
            pass

    with handoff.generation():
        with mock.patch.object(submodule_base, "rmtree", side_effect=advance_without_deleting):
            with pytest.raises(StaleSubmoduleGenerationError):
                submodule.remove(module=True, configuration=True, force=True)

    # The module-ledger phase ran, but the stale gitlink ledger was not allowed
    # to rewrite the parent index or the .gitmodules configuration.
    assert _entry_hexsha(parent.index, "module") == gitlink_before
    assert (Path(parent.working_tree_dir) / ".gitmodules").is_file()


# } END rollback and recovery


# { Pin parity: with no generation active, behaviour is unmanaged.


def test_unmanaged_add_records_no_generation(tmp_path):
    with Repo.init(tmp_path / "pin-source") as source, Repo.init(tmp_path / "pin-parent") as parent:
        source.git.symbolic_ref("HEAD", "refs/heads/master")
        _commit_file(source, "pin\n")
        readme = _wpath(parent, "README")
        readme.write_text("p\n")
        parent.index.add(["README"])
        parent.index.commit("init")

        handoff = submodule_handoff(parent)
        sm = parent.create_submodule("pin", "pin", source.working_tree_dir, branch="master")
        assert not handoff.active
        assert handoff.landings == ()
        assert parent.index.entry_key("pin", 0) in parent.index.entries
        assert (Path(parent.working_tree_dir) / ".gitmodules").is_file()

        # deinit and configuration removal likewise stay unmanaged without an
        # open generation (force-removal of the tree is exercised on platforms
        # without the Windows file-locking SkipTest wrapper).
        with sm.module():
            sm.deinit(force=True)
            assert handoff.landings == ()
            sm.remove(module=False, configuration=True)
            assert handoff.landings == ()
            assert parent.index.entry_key("pin", 0) not in parent.index.entries


def test_unmanaged_update_and_root_update_record_no_generation(local):
    parent, submodule, module = local.parent, local.submodule, local.module
    handoff = submodule_handoff(parent)

    module.head.reset("HEAD~1", index=True, working_tree=True)
    submodule.update(no_fetch=True)
    assert handoff.landings == ()

    previous = parent.head.commit
    RootModule(parent).update(previous_commit=previous, recursive=False, no_fetch=True)
    assert handoff.landings == ()
    assert not handoff.active


# } END pin parity
