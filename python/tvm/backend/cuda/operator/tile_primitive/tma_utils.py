# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""TMA (Tensor Memory Accelerator) utilities for CUDA op dispatches."""

import copy
import functools
import operator
from enum import Enum

import tvm
from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.tirx.layout import ComposeLayout, Layout, S, SwizzleLayout, TileLayout


class SwizzleMode(Enum):
    """The swizzle mode of the TMA."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B_ATOM = 1
    SWIZZLE_64B_ATOM = 2
    SWIZZLE_128B_ATOM = 3


def mma_atom_layout(dtype: str, swizzle_mode: SwizzleMode | int) -> SwizzleLayout:
    """Generate the MMA-compatible shared-memory atom layout."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    return SwizzleLayout(
        per_element=(128 // bits).bit_length() - 1, swizzle_len=swizzle_mode.value, atom_len=3
    )


def mma_atom_shape(dtype: str, swizzle_mode: SwizzleMode | int, shape: list[int] | None = None):
    """Generate the MMA-compatible shared-memory atom shape."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    atom_shape = {
        SwizzleMode.SWIZZLE_32B_ATOM: [8, 256],
        SwizzleMode.SWIZZLE_64B_ATOM: [8, 512],
        SwizzleMode.SWIZZLE_128B_ATOM: [8, 1024],
    }[swizzle_mode]
    atom_shape[-1] //= bits
    if shape is None:
        return atom_shape
    atom_shape = [1] * (len(shape) - len(atom_shape)) + atom_shape
    return atom_shape


def mma_shared_layout(dtype: str, swizzle_mode: SwizzleMode | int, shape) -> Layout:
    """Generate the MMA-compatible shared-memory layout for shape and dtype.

    It uses a default tiling strategy to tile the TMA atom layout into the shared memory.
    """
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return TileLayout(S[tuple(shape)]).canonicalize()
    atom_shape = mma_atom_shape(dtype, swizzle_mode, shape)
    layout = mma_atom_layout(dtype, swizzle_mode)
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


def mma_shared_layout_k_major(dtype: str, swizzle_mode: SwizzleMode | int, shape) -> Layout:
    """Generate a K-major shared layout for a logical ``[..., K, MN]`` region.

    ``mma_shared_layout`` makes the last logical axis contiguous.  This helper
    instead transposes the swizzle atom's two matrix axes, making the
    penultimate K axis contiguous while preserving the caller's logical
    indexing.  Leading axes are tiled independently, which permits staged
    buffers such as ``[stage, K, MN]``.  It is the layout expected by a Hopper
    WGMMA B descriptor with its transpose/major-mode bit clear.
    """

    if len(shape) < 2:
        raise ValueError(f"K-major MMA shared layout requires at least 2D, got {shape}")
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        leading_stride = shape[-2] * shape[-1]
        strides = [
            leading_stride * functools.reduce(operator.mul, shape[i + 1 : -2], 1)
            for i in range(len(shape) - 2)
        ]
        return TileLayout(S[tuple(shape) : (*strides, 1, shape[-2])]).canonicalize()
    base_shape = mma_atom_shape(dtype, swizzle_mode)
    matrix_atom_shape = [base_shape[1], base_shape[0]]
    transpose_tile = TileLayout(S[tuple(matrix_atom_shape) : (1, matrix_atom_shape[0])])
    atom = ComposeLayout(mma_atom_layout(dtype, swizzle_mode), transpose_tile)
    atom_shape = [1] * (len(shape) - 2) + matrix_atom_shape
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return atom.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


# Backward-compatible aliases kept during the alloc_mma migration.
tma_atom_layout = mma_atom_layout
tma_atom_shape = mma_atom_shape
tma_shared_layout = mma_shared_layout


def tma_atom_compatible(dst_shape, dst_st, dst_extent, atom_shape):
    """Check if the copy region in dst is compatible with the TMA atom shape."""
    analyzer = Analyzer()
    for i, _ in enumerate(dst_st):
        if any(
            not analyzer.can_prove_equal(x % atom_shape[i], 0)
            for x in [dst_shape[i], dst_st[i], dst_extent[i]]
        ):
            return False
    return True


def get_swizzle_mode_from_layout(layout: Layout) -> SwizzleMode | None:
    """Extract swizzle mode from a shared memory layout."""
    if isinstance(layout, ComposeLayout):
        swizzle = layout.swizzle  # SwizzleLayout is named 'swizzle' in ComposeLayout
        swizzle_len = swizzle.swizzle_len
    elif isinstance(layout, SwizzleLayout):
        swizzle_len = layout.swizzle_len
    elif isinstance(layout, TileLayout):
        # TileLayout without SwizzleLayout means no swizzle (mode 0)
        return SwizzleMode.SWIZZLE_NONE
    else:
        return None

    # Map swizzle_len to SwizzleMode
    return {
        0: SwizzleMode.SWIZZLE_NONE,
        1: SwizzleMode.SWIZZLE_32B_ATOM,
        2: SwizzleMode.SWIZZLE_64B_ATOM,
        3: SwizzleMode.SWIZZLE_128B_ATOM,
    }.get(swizzle_len)


def get_mma_smem_descriptor_params(buf, buf_region, dtype, is_transposed):
    """Derive MMA shared-memory descriptor parameters from a sliced layout.

    The logical transpose flag identifies which buffer dimension is K.  The
    physical swizzle match then returns ``(swizzle, ldo, sdo, mn_major)`` for
    use by either tcgen05 or WGMMA.  Sub-atom contiguous slices are accepted
    only when their start is 16-byte aligned; the descriptor describes the
    enclosing atom while the caller selects the exact tile address.
    """

    if buf.layout is None:
        raise ValueError("MMA shared-memory descriptor requires an explicit buffer layout")

    analyzer = Analyzer()
    region = list(buf_region.region)

    def _match(slice_layout, shape_2d):
        def _try_atom(mode, atom, atom_shape):
            if any(s % a != 0 for s, a in zip(shape_2d, atom_shape)):
                return None
            atom_size = functools.reduce(operator.mul, atom_shape, 1)
            tiler = atom.is_tile_inner(slice_layout, shape_2d, atom_shape)
            if tiler is None:
                return None
            tiler_shape = [s // a for s, a in zip(shape_2d, atom_shape)]
            tiler_grouped, _ = tiler.canonicalize().group(tiler_shape)
            elem_per_16b = 128 // tvm.DataType(dtype).bits

            def _atom_off(dim):
                if int(dim.extent) == 1:
                    return 0
                return (dim.stride * atom_size) // elem_per_16b

            ldo = _atom_off(tiler_grouped.shard[-1])
            sdo = _atom_off(tiler_grouped.shard[-2])
            return mode, ldo, sdo

        for mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            swizzle_atom = mma_atom_layout(dtype, mode)
            base_shape = mma_atom_shape(dtype, mode)
            swapped_shape = [base_shape[1], base_shape[0]]
            mn_tile = TileLayout(S[tuple(swapped_shape) : (1, swapped_shape[0])])
            mn_atom = ComposeLayout(swizzle_atom, mn_tile)
            if is_transposed:
                candidates = [
                    (False, mn_atom, swapped_shape),
                    (True, swizzle_atom, base_shape),
                ]
            else:
                candidates = [
                    (False, swizzle_atom, base_shape),
                    (True, mn_atom, swapped_shape),
                ]

            for is_mn_major, atom, atom_shape in candidates:
                result = _try_atom(mode, atom, atom_shape)
                if result is not None:
                    swizzle, ldo, sdo = result
                    if is_mn_major != is_transposed:
                        ldo, sdo = sdo, ldo
                    return swizzle, ldo, sdo, is_mn_major
        return None

    cax = len(region) - 1
    elem_per_16b = 128 // DataType(dtype).bits
    phys_mode = get_swizzle_mode_from_layout(buf.layout)
    desc_region = list(region)
    if phys_mode in (
        SwizzleMode.SWIZZLE_128B_ATOM,
        SwizzleMode.SWIZZLE_64B_ATOM,
        SwizzleMode.SWIZZLE_32B_ATOM,
    ):
        atom_inner = mma_atom_shape(dtype, phys_mode)[-1]
        contig = int(region[cax].extent)
        rounded = ((contig + atom_inner - 1) // atom_inner) * atom_inner
        if rounded != contig:
            if not analyzer.can_prove_equal(tvm.tirx.floormod(region[cax].min, elem_per_16b), 0):
                raise ValueError(
                    f"MMA shared-memory slice start {region[cax].min} is not "
                    f"16-byte aligned ({elem_per_16b} elements for {dtype})"
                )
            desc_region[cax] = tvm.ir.Range.from_min_extent(0, rounded)

    slice_layout = buf.layout.slice(buf.shape, desc_region)
    shape_2d = [int(r.extent) for r in desc_region if int(r.extent) != 1]
    if len(shape_2d) != 2:
        raise ValueError(
            "MMA shared-memory descriptor expects exactly two non-unit dimensions, "
            f"got {[int(r.extent) for r in desc_region]}"
        )
    result = _match(slice_layout, shape_2d)
    if result is not None:
        return result

    hint = ""
    if phys_mode in (
        SwizzleMode.SWIZZLE_128B_ATOM,
        SwizzleMode.SWIZZLE_64B_ATOM,
        SwizzleMode.SWIZZLE_32B_ATOM,
    ):
        atom_inner = mma_atom_shape(dtype, phys_mode)[-1]
        atom_bytes = atom_inner * DataType(dtype).bits // 8
        hint = (
            f" Physical layout is {phys_mode.name} with a contiguous "
            f"{atom_inner}-element/{atom_bytes}-byte atom."
        )
    raise ValueError(
        f"No MMA shared-memory descriptor matches region shape {shape_2d} for dtype {dtype}.{hint}"
    )
