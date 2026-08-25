"""
Turn an ordinary NumPy processing function into a chunk-wise Dask one.

Why this exists
---------------
The worker converts a lazy input to NumPy for any function that does not
accept ``_source_filepath`` (``_file_selector.py``), so most functions
materialise the whole stack and their peak tracks the input — 30 of the 39
audited functions measure ~2.0x growth.

The plumbing to avoid that already exists on both sides: the worker keeps the
input lazy for functions that opt in, and streams a lazy *result* back to disk
block by block.  The only missing piece is the function body, which returns a
dense array in between.  ``@chunked`` supplies that piece without rewriting
any body: it maps the existing NumPy function over blocks of the lazy input
and hands back a Dask array.

Why no overlap is needed
------------------------
These functions are already written to treat the leading axes as independent
— that is exactly what their ``dimension_order`` branching expresses.  So
blocks are cut along the *leading* axes only and the trailing spatial axes are
never split.  A YX plane is small even on the reference acquisition
(2720x2720 uint16 = 15 MB), so splitting T/C/Z alone bounds memory with room
to spare.

That restriction is what makes this safe: because a block always contains
whole planes (or whole volumes), the body sees exactly the same array shapes
it would have seen densely, and the result is byte-identical to the dense
path.  No halo, no boundary artefacts, no re-tuning of any algorithm.  A
function that genuinely couples the axes it is chunked along must not use
this decorator — see ``trailing_whole``.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np

# Target size of one materialised block.  Dask's threaded scheduler holds
# roughly one block per core, so this is multiplied by the worker count --
# keep it well under _STREAM_BLOCK_BYTES (256 MB) for that reason.
_DEFAULT_BLOCK_BYTES = 64 * 1024 * 1024


def _is_lazy(array) -> bool:
    return hasattr(array, "chunks") and hasattr(array, "compute")


def _signature_with_source_filepath(signature):
    """
    `signature` plus a keyword-only ``_source_filepath``, if absent.

    The worker decides whether a function may keep its input lazy by looking
    for that parameter, and ``inspect.signature`` follows ``__wrapped__`` (set
    by ``functools.wraps``), so a decorator that leaves ``__signature__`` alone
    advertises the undecorated body and never opts in.  The parameter has to be
    inserted before any ``**kwargs`` rather than appended, since a variadic
    keyword must stay last -- appending raises ``ValueError`` from ``replace``.
    """
    if "_source_filepath" in signature.parameters:
        return signature

    params = list(signature.parameters.values())
    insert_at = len(params)
    for i, param in enumerate(params):
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            insert_at = i
            break
    params.insert(
        insert_at,
        inspect.Parameter(
            "_source_filepath",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
        ),
    )
    return signature.replace(parameters=params)


def _block_chunks(shape, itemsize, keep, block_bytes):
    """
    Chunk spec keeping the trailing `keep` axes whole and splitting leading
    ones until a block fits `block_bytes`.

    Grows the innermost leading axis first so blocks stay contiguous in the
    source layout; anything further out stays at 1.
    """
    unit = int(np.prod(shape[len(shape) - keep :], dtype=np.int64)) * itemsize
    chunks = [1] * (len(shape) - keep) + [-1] * keep
    if not chunks or unit >= block_bytes:
        return tuple(chunks)

    # Aim for several blocks, not just one that happens to fit the budget.
    # With a single leading axis (keep=3 on TZYX) the budget alone would grow
    # that axis to cover the whole array, producing one block -- which
    # materialises everything at once *and* leaves Dask one task to run on
    # however many cores are free.  Targeting >= 8 blocks keeps the scheduler
    # fed; the byte budget still caps the block on genuinely large inputs.
    total = int(np.prod(shape, dtype=np.int64)) * itemsize
    budget = min(block_bytes, max(unit, total // 8))

    n_units = max(1, int(budget // max(unit, 1)))
    axis = len(shape) - keep - 1
    if axis >= 0:
        chunks[axis] = min(int(shape[axis]), n_units)
    return tuple(chunks)


def plane_wise_only(per_plane_orders):
    """
    A ``trailing_whole`` resolver for filters whose behaviour depends on the
    dimension_order hint.

    Filters like Gaussian blur act per YX plane for hints such as ``TZYX``
    but blur the whole volume for ``ZYX`` — and for ``Auto`` they couple
    every axis they are given.  Splitting a coupled axis would change the
    result, so anything not explicitly per-plane resolves to "keep the whole
    array", which makes ``@chunked`` fall back to the dense path.
    """

    def resolve(dimension_order, ndim):
        order = str(dimension_order or "").upper()
        if order in per_plane_orders and len(order) == ndim:
            return 2
        return ndim

    return resolve


def lazy_capable(body):
    """
    Opt a function into a lazy input without wrapping its body at all.

    For a *reduction* along an axis that `@chunked` would split — a Z
    projection, say — mapping over blocks is the wrong tool: the answer
    combines blocks rather than being computed within one.  Dask already
    does that correctly and with bounded memory via a tree reduction, so
    such a body only needs to be written in terms that dispatch to Dask
    (``np.max(image, axis=0)`` does) and to be told it may receive a lazy
    array.  This decorator is only the second half of that.
    """
    signature = inspect.signature(body)

    @functools.wraps(body)
    def wrapper(*args, **kwargs):
        kwargs.pop("_source_filepath", None)
        return body(*args, **kwargs)

    wrapper.__signature__ = _signature_with_source_filepath(signature)
    return wrapper


def independent_leading(independent="TC"):
    """
    A ``trailing_whole`` resolver matching ``_iter_dimension_blocks``.

    That helper treats T and C as independent but deliberately keeps Z with
    Y and X, so a 3D object spanning several Z slices gets one label rather
    than one per slice.  Splitting Z would reintroduce exactly that bug, so
    only the leading *run* of T/C axes may be split: ``TZYX`` -> 3 (split T
    alone), ``TCZYX`` -> 3, ``TYX`` -> 2.

    An order whose first axis is not T or C (``ZCYX``, ``ZYX``) resolves to
    the full rank, i.e. the dense fallback -- the C in ``ZCYX`` is
    independent but it is not *leading*, and blocks are cut from the front.
    """

    def resolve(dimension_order, ndim):
        order = str(dimension_order or "").upper()
        if len(order) != ndim or not order.endswith("YX"):
            return ndim
        if set(order) - set("TCZYX"):
            return ndim
        leading = 0
        for axis in order:
            if axis not in independent:
                break
            leading += 1
        return ndim - leading

    return resolve


def chunked(trailing_whole=2, block_bytes=_DEFAULT_BLOCK_BYTES):
    """
    Map a NumPy processing function over blocks of a lazy input.

    Parameters
    ----------
    trailing_whole : int | callable
        How many trailing axes must stay in one piece — i.e. the axes the
        function actually couples.  ``2`` (YX) for anything that works a
        plane at a time; ``3`` (ZYX) for anything that couples Z, such as
        connected-component labelling of a 3D volume.  Pass a callable
        ``(dimension_order, ndim) -> int`` where it depends on the hint.
        **Getting this too small silently corrupts results** (objects split
        at a block edge, a background estimated from part of a volume), so
        it errs upward when the hint cannot be resolved.
    block_bytes : int
        Target materialised bytes per block.

    A dense input is passed straight through to the undecorated body, so the
    small-input path and all existing tests are unaffected.
    """

    def decorate(body):
        signature = inspect.signature(body)
        image_param = next(iter(signature.parameters))

        @functools.wraps(body)
        def wrapper(image, *args, **kwargs):
            kwargs.pop("_source_filepath", None)

            bound = signature.bind_partial(image, *args, **kwargs)
            bound.apply_defaults()
            call_kwargs = dict(bound.arguments)
            call_kwargs.pop(image_param, None)

            if not _is_lazy(image):
                return body(image, **call_kwargs)

            import dask.array as da

            keep = trailing_whole
            if callable(keep):
                keep = keep(call_kwargs.get("dimension_order"), image.ndim)
            keep = min(max(int(keep), 1), int(image.ndim))

            if keep >= image.ndim:
                # Nothing may be split: honour that rather than chunk wrongly.
                return body(np.asarray(image), **call_kwargs)

            array = image.rechunk(
                _block_chunks(
                    tuple(int(s) for s in image.shape),
                    np.dtype(image.dtype).itemsize,
                    keep,
                    block_bytes,
                )
            )

            # Run the body on the first block to learn the output dtype, and
            # to fail loudly here rather than inside a Dask task.  The block
            # is bounded by block_bytes, so this is cheap.
            first = array[
                tuple(slice(0, c[0]) for c in array.chunks)
            ].compute()
            probe = np.asarray(body(first, **call_kwargs))

            n_leading = array.ndim - keep
            if (
                probe.ndim != first.ndim
                or probe.shape[:n_leading] != first.shape[:n_leading]
            ):
                # Either an axis vanished (a projection) or a leading axis
                # changed size — both mean the body combines or removes the
                # very axes blocks are cut along, so mapping block-for-block
                # would mis-assemble the result.  A reduction along a split
                # axis wants Dask's own tree reduction, not map_blocks.
                return body(np.asarray(image), **call_kwargs)

            mapped = lambda block: np.asarray(  # noqa: E731
                body(block, **call_kwargs)
            )
            out_trailing = probe.shape[n_leading:]

            if out_trailing == first.shape[n_leading:]:
                return da.map_blocks(
                    mapped,
                    array,
                    dtype=probe.dtype,
                    meta=np.empty((0,) * array.ndim, dtype=probe.dtype),
                )

            # The trailing extent changed — a per-plane resize.  Every block
            # holds the whole spatial plane and resizes it to the same
            # target, so the output chunking is just the input's leading
            # chunks with the new trailing sizes.
            return da.map_blocks(
                mapped,
                array,
                dtype=probe.dtype,
                chunks=tuple(array.chunks[:n_leading])
                + tuple((int(size),) for size in out_trailing),
                meta=np.empty((0,) * array.ndim, dtype=probe.dtype),
            )

        wrapper.__signature__ = _signature_with_source_filepath(signature)
        wrapper.__chunked_body__ = body
        return wrapper

    return decorate
