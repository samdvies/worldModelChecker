"""Guard for failure class A: embedded `uv run python -c "..."` snippets in
scripts/aws/*.sh are not covered by pytest collection (they are plain shell
string literals), so a signature drift like `enc.encode(clip.frames)` vs the
real `encode(clip: Clip)` only surfaces on the GPU box. This statically
extracts every such snippet, parses it, and for every call whose callee
resolves to a real physics_auditor callable, checks the call's arity
against `inspect.signature` AND (see `_infer_expr_type` / `_class_has_field`
below) does a best-effort TYPE/shape check: if a local variable is already
known to be exactly the type a parameter is annotated with, but the
argument passed is `that_local.some_attr` rather than `that_local` itself,
that is flagged even though the arity matches -- `clip.frames` and `clip`
are both exactly one positional argument, so arity alone is blind to this,
which is precisely why the original `enc.encode(clip.frames)` vs
`encode(clip: Clip)` bug slipped past a pure-arity check.
"""
import ast
import glob
import importlib
import inspect
import re
import typing
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AWS_SCRIPTS = sorted(glob.glob(str(REPO_ROOT / "scripts" / "aws" / "*.sh")))

# Matches: uv run python -c "<body>"   (body may span multiple lines, ends
# at the first line consisting of just a closing double-quote).
_SNIPPET_RE = re.compile(
    r'uv run python -c "\n(.*?)\n"',
    re.DOTALL,
)


def _extract_snippets(sh_path: str) -> list[str]:
    text = Path(sh_path).read_text(encoding="utf-8")
    return _SNIPPET_RE.findall(text)


def _all_snippets() -> list[tuple[str, str]]:
    out = []
    for sh_path in AWS_SCRIPTS:
        for i, snippet in enumerate(_extract_snippets(sh_path)):
            out.append((f"{sh_path}#{i}", snippet))
    return out


SNIPPETS = _all_snippets()


def test_at_least_one_embedded_snippet_found():
    """Sanity check the extractor itself still finds remote_job.sh's known
    embedded smoke-test snippets -- if this goes to 0, the regex silently
    stopped matching and the whole guard below would pass vacuously."""
    assert len(SNIPPETS) >= 2, (
        f"expected >=2 embedded python snippets across {AWS_SCRIPTS}, found "
        f"{len(SNIPPETS)} -- did remote_job.sh's snippet formatting change?"
    )


def _resolve_attr_chain(names: list[str]):
    """names = ['physics_auditor', 'models', 'pretrained', 'DINOv2Encoder'].
    Try importing progressively longer module paths, then resolve remaining
    names as attributes. Returns the resolved object, or None if any step
    fails (i.e. not confidently resolvable -- skip rather than false-flag)."""
    obj = None
    consumed = 0
    for i in range(len(names), 0, -1):
        mod_name = ".".join(names[:i])
        try:
            obj = importlib.import_module(mod_name)
            consumed = i
            break
        except ImportError:
            continue
    if obj is None:
        return None
    for name in names[consumed:]:
        if not hasattr(obj, name):
            return None
        obj = getattr(obj, name)
    return obj


def _dotted_name(node: ast.AST) -> list[str] | None:
    """AST Attribute/Name chain -> list of names, root-first. None if the
    chain contains anything other than Name/Attribute (e.g. a call)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _return_annotation(func):
    """`inspect.signature(func).return_annotation`, but with string /
    forward-ref annotations resolved via `typing.get_type_hints` (handles
    e.g. `def load(self) -> "DINOv2Encoder":` self-referencing forward
    refs, which `inspect.signature` alone leaves as a bare string that
    `_resolve_annotation_class` can't match against a real class)."""
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    if "return" in hints:
        return hints["return"]
    try:
        return inspect.signature(func).return_annotation
    except (TypeError, ValueError):
        return inspect.Signature.empty


def _resolve_annotation_class(ann):
    """Best-effort: resolve a signature annotation to a single concrete
    class. Handles plain classes directly and simple one-arg generics like
    `Iterator[Clip]` / `Iterable[Clip]` (unwraps to the element type, which
    is what a variable bound via `next(iter(...))` actually holds). Returns
    None when it can't confidently resolve to exactly one class (empty
    annotation, unions, bare `Iterator` with no arg, etc.) -- callers must
    treat None as "skip, don't false-flag"."""
    if isinstance(ann, type):
        return ann
    args = typing.get_args(ann)
    if len(args) == 1 and isinstance(args[0], type):
        return args[0]
    return None


def _class_has_field(cls: type, name: str) -> bool:
    """True if `cls` declares `name` as a dataclass field or a plain
    attribute. Plain `hasattr(cls, name)` misses dataclass fields (they're
    only set on instances, not the class object)."""
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields is not None and name in fields:
        return True
    return hasattr(cls, name)


def _resolve_call_target(call_node: ast.Call, local_types: dict):
    """Resolve a Call node's callee to a real function/method object.
    Handles:
      - plain dotted Name/Attribute chains, e.g.
        `physics_auditor.models.pretrained.DINOv2Encoder` (via
        `_resolve_attr_chain`), or a chain rooted at a local already
        tracked in `local_types` (a class -> instance method lookup, or an
        already-resolved free function bound via an `ImportFrom`), e.g.
        `enc.encode(...)` where `enc` is-a `DINOv2Encoder`.
      - one level of method chaining, e.g. `DINOv2Encoder().load()`: the
        callee's `.value` is itself a Call, so its type is inferred via
        `_infer_expr_type` and the method looked up on that type.
    Returns None when not confidently resolvable (skip, don't false-flag)."""
    func = call_node.func
    chain = _dotted_name(func)

    if chain is None:
        # func.value is itself a Call, e.g. `DINOv2Encoder().load` --
        # method chaining. Only handled when func itself is an Attribute
        # (a bare call-of-a-call, `f()()`, isn't).
        if not isinstance(func, ast.Attribute):
            return None
        owner_type = _infer_expr_type(func.value, local_types)
        if owner_type is None or not hasattr(owner_type, func.attr):
            return None
        return getattr(owner_type, func.attr)

    if len(chain) == 1:
        maybe_local = local_types.get(chain[0])
        if inspect.isfunction(maybe_local) or inspect.ismethod(maybe_local):
            # already-resolved free function bound via an ImportFrom, e.g.
            # `train_clips` -- no need to re-resolve via import machinery.
            return maybe_local
        return _resolve_attr_chain(chain)

    root = local_types.get(chain[0])
    if root is not None:
        obj = root
        for attr in chain[1:]:
            if not hasattr(obj, attr):
                return None
            obj = getattr(obj, attr)
        return obj
    return _resolve_attr_chain(chain)


def _infer_expr_type(node: ast.AST, local_types: dict):
    """Best-effort static type inference for the handful of shapes these
    embedded snippets actually use:
      - `Name` already bound in `local_types` (e.g. `enc` after
        `enc = DINOv2Encoder()`, or `clip` after `clip = simulate(cfg)`).
      - a `Call` to a resolvable function/method (via
        `_resolve_call_target`, which also handles one level of method
        chaining like `DINOv2Encoder().load()`): uses its return
        annotation (unwrapped through `_resolve_annotation_class`, with
        forward-ref strings resolved via `_return_annotation`), e.g.
        `simulate(cfg) -> Clip`, `train_clips() -> Iterator[Clip]`, or
        `DINOv2Encoder().load() -> "DINOv2Encoder"`.
      - `iter(X)` / `next(X)`: treated as identity over the inferred type
        of `X` -- the generic unwrap above already produces the element
        type, not the container type, so neither call changes it further.
      - a bare constructor call `Cls()` where `Cls` is a class already
        bound in `local_types` (e.g. via an `ImportFrom`): the resulting
        instance "is-a" `Cls`.
    Returns None when not confidently inferable (skip, don't false-flag)."""
    if isinstance(node, ast.Name):
        t = local_types.get(node.id)
        return t if isinstance(t, type) else None
    if not isinstance(node, ast.Call):
        return None

    chain = _dotted_name(node.func)
    if chain in (["iter"], ["next"]) and len(node.args) == 1:
        return _infer_expr_type(node.args[0], local_types)

    if chain and len(chain) == 1:
        maybe_cls = local_types.get(chain[0])
        if isinstance(maybe_cls, type):
            return maybe_cls  # bare constructor call, e.g. DINOv2Encoder()

    target = _resolve_call_target(node, local_types)
    if target is None or not (inspect.isfunction(target) or inspect.ismethod(target)):
        return None
    return _resolve_annotation_class(_return_annotation(target))


@pytest.mark.parametrize("label,snippet", SNIPPETS, ids=[s[0] for s in SNIPPETS])
def test_embedded_snippet_parses_and_calls_resolve(label, snippet):
    tree = ast.parse(snippet, filename=label)

    # Track simple local variable -> resolved-object bindings so we can
    # resolve calls like `enc.encode(clip)` where `enc = DINOv2Encoder()`,
    # and (for the type check below) `clip`'s real type after
    # `clip = simulate(cfg)` or `clip = next(iter(train_clips()))`.
    local_types: dict[str, object] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = importlib.import_module(node.module)
            for alias in node.names:
                if hasattr(mod, alias.name):
                    local_types[alias.asname or alias.name] = getattr(mod, alias.name)

        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            inferred = _infer_expr_type(node.value, local_types)
            if inferred is not None:
                local_types[node.targets[0].id] = inferred

        if isinstance(node, ast.Call):
            target = _resolve_call_target(node, local_types)

            if target is None or not (inspect.isfunction(target) or inspect.ismethod(target)):
                continue  # not confidently resolvable to a real callable -- skip

            try:
                sig = inspect.signature(target)
            except (TypeError, ValueError):
                continue

            n_args = len(node.args)
            n_kwargs = {kw.arg for kw in node.keywords if kw.arg}
            params = list(sig.parameters.values())
            # drop 'self' for bound methods retrieved off an instance-ish object
            if params and params[0].name == "self" and inspect.isfunction(target):
                params = params[1:]

            call_desc = ast.unparse(node.func) if hasattr(ast, "unparse") else target

            # Type/shape check (arity-blind bug class): if a positional
            # argument is `some_local.attr`, and `some_local` is already
            # known to be EXACTLY the type the parameter is annotated
            # with, then passing `.attr` off it instead of the local
            # itself is (near-)certainly a mistake -- the exact shape of
            # `enc.encode(clip.frames)` vs `encode(clip: Clip)`.
            for arg_node, param in zip(node.args, params):
                if not isinstance(arg_node, ast.Attribute) or not isinstance(arg_node.value, ast.Name):
                    continue
                ann_cls = _resolve_annotation_class(param.annotation)
                if ann_cls is None:
                    continue
                base_type = local_types.get(arg_node.value.id)
                if base_type is not ann_cls:
                    continue
                if not _class_has_field(ann_cls, arg_node.attr):
                    continue
                assert False, (
                    f"{label}: call to {call_desc} passes "
                    f"`{arg_node.value.id}.{arg_node.attr}` for parameter "
                    f"`{param.name}` annotated `{ann_cls.__name__}`, but "
                    f"`{arg_node.value.id}` is already a `{ann_cls.__name__}` "
                    f"-- likely meant to pass `{arg_node.value.id}` itself, "
                    f"not `.{arg_node.attr}` off it (this is the exact shape "
                    f"of the historical class-A bug: `enc.encode(clip.frames)` "
                    f"vs `encode(clip: Clip)`, which pure-arity checking "
                    f"cannot see since both count as one positional arg)"
                )

            required_positional = [
                p for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
                and p.name not in n_kwargs
            ]
            has_var_positional = any(p.kind == p.VAR_POSITIONAL for p in params)
            max_positional = sum(
                1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            )

            assert n_args >= len(required_positional), (
                f"{label}: call to {call_desc} passes {n_args} positional "
                f"args but {len(required_positional)} are required by "
                f"{target} -- signature is {sig}"
            )
            assert has_var_positional or n_args <= max_positional, (
                f"{label}: call to {call_desc} passes {n_args} positional "
                f"args but at most {max_positional} are accepted by "
                f"{target} -- signature is {sig}"
            )
