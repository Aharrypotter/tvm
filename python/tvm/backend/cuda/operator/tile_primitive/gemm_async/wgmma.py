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

"""SM90a WGMMA lowering for the asynchronous ``gemm_async`` tile op."""

import tvm
from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import PrimFunc
from tvm.tirx.layout import (
    wgmma_a_register_layout,
    wgmma_accumulator_layout,
)
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import cuda_arch_matches
from ..tma_utils import get_mma_smem_descriptor_params

_WGMMA_N = (16, 32, 64, 128)
_WGMMA_INSTR_N = 64
_WGMMA_K = 16
_WGMMA_INPUT_DTYPES = ("float16", "bfloat16")


def _full_warpgroup(op: TilePrimitiveCall, sctx: DispatchContext):
    """Require one complete, un-narrowed 128-thread warpgroup."""

    if not sctx.is_warpgroup:
        return False, f"expected warpgroup scope, got {sctx.scope_kind}"
    full = {"laneid": 32, "wid_in_wg": 4}
    if set(sctx.intra) != set(full):
        return (
            False,
            f"warpgroup active-set axes are {sorted(sctx.intra)}, need {sorted(full)}",
        )
    for axis, rng in sctx.intra.items():
        extent, offset = int(rng[0]), int(rng[1])
        if extent != full[axis] or offset != 0:
            return False, (
                f"active {axis} is [{offset}, {offset + extent}), need full [0, {full[axis]})"
            )
    return True, None


def _const_bool(analyzer, expr, name):
    if isinstance(expr, bool):
        return expr
    value = analyzer.simplify(expr)
    try:
        return bool(int(value.value))
    except (AttributeError, TypeError, ValueError):
        fail(f"WGMMA needs a static {name} flag, got {expr}")


def _const_int(analyzer, expr, name):
    if isinstance(expr, int):
        return expr
    value = analyzer.simplify(expr)
    try:
        return int(value.value)
    except (AttributeError, TypeError, ValueError):
        fail(f"WGMMA needs a static integer {name}, got {expr}")


def _mat_extent_info(analyzer, region, name):
    dims = [
        (i, int(r.extent))
        for i, r in enumerate(region.region)
        if not analyzer.can_prove_equal(r.extent, 1)
    ]
    if len(dims) != 2:
        fail(f"WGMMA expects two non-unit dimensions for {name}, got {dims}")
    return dims


def _require_layout(actual, expected, name):
    if actual is None:
        fail(f"WGMMA requires an explicit {name} layout")
    try:
        actual = actual.canonicalize()
        expected = expected.canonicalize()
        tvm.ir.assert_structural_equal(actual, expected)
    except (AssertionError, ValueError, RuntimeError) as err:
        fail(f"WGMMA {name} fragment layout mismatch: {err}")


def _layout_equal(actual, expected):
    if actual is None:
        return False
    try:
        tvm.ir.assert_structural_equal(actual.canonicalize(), expected.canonicalize())
        return True
    except (AssertionError, ValueError, RuntimeError):
        return False


def _require_accumulator_region_layout(analyzer, buffer, region, matrix_dims, M, N, dtype, name):
    """Accept one m64 accumulator tile, including a wgid-sharded parent."""

    expected = wgmma_accumulator_layout(M, N, dtype)
    if buffer.layout is None:
        fail(f"WGMMA requires an explicit {name} layout")
    sliced = buffer.layout.slice(buffer.shape, list(region.region))
    if _layout_equal(sliced, expected):
        return

    # A resident m128 (or larger) fragment can be sharded over adjacent
    # warpgroups.  A fixed m64 region then retains a wgid offset after layout
    # slicing even though its per-thread register order is exactly the
    # instruction order.  Validate the parent layout plus aligned slice
    # explicitly instead of rejecting that harmless thread-axis offset.
    m_dim, n_dim = matrix_dims[0][0], matrix_dims[1][0]
    try:
        full_m, full_n = int(buffer.shape[m_dim]), int(buffer.shape[n_dim])
        m_min = int(analyzer.simplify(region.region[m_dim].min))
        n_min = int(analyzer.simplify(region.region[n_dim].min))
    except (TypeError, ValueError):
        fail(f"WGMMA {name} needs a static aligned register region")
    full_expected = wgmma_accumulator_layout(full_m, full_n, dtype)
    if (
        M != 64
        or m_min % 64
        or n_min % 16
        or n_min + N > full_n
        or not _layout_equal(buffer.layout, full_expected)
    ):
        fail(f"WGMMA {name} fragment layout mismatch for region {region.region}")
    for dim, rng in enumerate(region.region):
        if dim not in (m_dim, n_dim) and not analyzer.can_prove_equal(rng.extent, 1):
            fail(f"WGMMA {name} has a non-unit outer region at dimension {dim}")


def _accumulator_region_local_offset(analyzer, region, matrix_dims, M, name):
    """Return the region start in the parent accumulator's logical register order."""

    n_dim = matrix_dims[1][0]
    try:
        n_min = int(analyzer.simplify(region.region[n_dim].min))
    except (TypeError, ValueError):
        fail(f"WGMMA {name} needs a static N-region offset")
    logical_offset = M * n_min
    if logical_offset % 128:
        fail(f"WGMMA {name} N-region offset {n_min} is not register aligned")
    return logical_offset // 128


def _require_full_region(analyzer, buffer, region, name):
    for dim, (shape, rng) in enumerate(zip(buffer.shape, region.region)):
        if not analyzer.can_prove_equal(rng.min, 0) or not analyzer.can_prove_equal(
            rng.extent, shape
        ):
            fail(
                f"WGMMA {name} register operand must cover its full buffer; "
                f"dimension {dim} has region [{rng.min}, {rng.min + rng.extent}) "
                f"for shape {shape}"
            )


def _descriptor_starts(region, k_dim, k_tile, n_dim=None, n_tile=0, instruction_n=_WGMMA_INSTR_N):
    starts = [r.min for r in region.region]
    starts[k_dim] = starts[k_dim] + k_tile * _WGMMA_K
    if n_dim is not None:
        starts[n_dim] = starts[n_dim] + n_tile * instruction_n
    return starts


def gemm_async_wgmma_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower one warpgroup ``gemm_async`` call to WGMMA SS or RS instructions.

    The accepted slice is deliberately explicit: BF16/FP16 inputs, FP32
    register accumulation, M=64, N in {16, 32, 64, 128}, and K decomposed into
    k16 instructions.  Logical N=16 and N=32 use one native instruction.
    By default, N=128 is decomposed into two m64n64 instructions.  An explicit
    ``wgmma_instruction_n`` configuration may select any supported native N.
    WGMMA group commit/wait remains caller-visible so a multi-op consumer can
    pipeline several tile calls into one group.
    """

    op_call = TilePrimitiveCall.downcast(op_call)
    if op_call.is_block_scaled:
        fail("WGMMA block-scaled gemm_async is not supported")

    C_region, A_region, B_region = op_call.output, op_call.lhs, op_call.rhs
    C, A, B = C_region.buffer, A_region.buffer, B_region.buffer
    C_scope, A_scope, B_scope = C.scope(), A.scope(), B.scope()
    if C_scope != "local":
        fail(f"WGMMA requires C in register (local) scope, got {C_scope}")
    if not B_scope.startswith("shared"):
        fail(f"WGMMA requires B in shared scope, got {B_scope}")
    if A_scope.startswith("shared"):
        operand_mode = "ss"
    elif A_scope == "local":
        operand_mode = "rs"
    else:
        fail(f"WGMMA requires A in shared or register (local) scope, got {A_scope}")

    requested_mode = op_call.config.get("operand_mode")
    if requested_mode is not None and requested_mode != operand_mode:
        fail(
            f"WGMMA operand_mode={requested_mode!r} does not match "
            f"A scope {A_scope!r} ({operand_mode})"
        )

    analyzer = Analyzer()
    trans_a = _const_bool(analyzer, op_call.transA, "transA")
    trans_b = _const_bool(analyzer, op_call.transB, "transB")
    accum = _const_bool(analyzer, op_call.accum, "accum")
    if operand_mode == "rs" and trans_a:
        fail("WGMMA RS requires transA=False; the register A form has no transA modifier")

    C_dims = _mat_extent_info(analyzer, C_region, "C")
    A_dims = _mat_extent_info(analyzer, A_region, "A")
    B_dims = _mat_extent_info(analyzer, B_region, "B")
    M, N = C_dims[0][1], C_dims[1][1]
    instruction_n = _const_int(
        analyzer,
        op_call.config.get("wgmma_instruction_n", min(_WGMMA_INSTR_N, N)),
        "wgmma_instruction_n",
    )
    if instruction_n not in _WGMMA_N:
        fail(f"WGMMA wgmma_instruction_n must be one of {_WGMMA_N}, got {instruction_n}")
    A_M, K = (A_dims[1][1], A_dims[0][1]) if trans_a else (A_dims[0][1], A_dims[1][1])
    B_K, B_N = (
        (B_dims[0][1], B_dims[1][1])
        if trans_b
        else (
            B_dims[1][1],
            B_dims[0][1],
        )
    )
    if M != 64:
        fail(f"WGMMA requires M=64, got {M}")
    if N not in _WGMMA_N:
        fail(f"WGMMA currently supports N in {_WGMMA_N}, got {N}")
    if N % instruction_n:
        fail(f"WGMMA logical N={N} must be divisible by wgmma_instruction_n={instruction_n}")
    if K < _WGMMA_K or K % _WGMMA_K:
        fail(f"WGMMA requires K divisible by {_WGMMA_K}, got {K}")
    if A_M != M or B_K != K or B_N != N:
        fail(f"WGMMA shape mismatch: C=({M},{N}), A logical=({A_M},{K}), B logical=({B_K},{B_N})")

    C_dtype, A_dtype, B_dtype = str(C.dtype), str(A.dtype), str(B.dtype)
    lhs_cast = op_call.config.get("lhs_cast")
    if lhs_cast is not None:
        if operand_mode != "rs":
            fail("WGMMA lhs_cast is supported only for the register A operand")
        if A_dtype != "float32":
            fail(f"WGMMA lhs_cast requires an FP32 register A operand, got {A_dtype}")
        if lhs_cast != B_dtype:
            fail(f"WGMMA lhs_cast={lhs_cast!r} must match the B dtype {B_dtype!r}")
        if lhs_cast != "bfloat16":
            fail(f"WGMMA lhs_cast currently supports only bfloat16, got {lhs_cast!r}")
    if C_dtype != "float32":
        fail(f"WGMMA requires FP32 accumulator, got {C_dtype}")
    input_dtype = lhs_cast if lhs_cast is not None else A_dtype
    if input_dtype not in _WGMMA_INPUT_DTYPES or B_dtype != input_dtype:
        fail(f"WGMMA requires matching float16/bfloat16 inputs, got A={A_dtype}, B={B_dtype}")

    _require_accumulator_region_layout(
        analyzer, C, C_region, C_dims, M, N, C_dtype, "C accumulator"
    )
    C_local_offset = _accumulator_region_local_offset(
        analyzer, C_region, C_dims, M, "C accumulator"
    )
    if operand_mode == "rs":
        if lhs_cast is not None:
            _require_accumulator_region_layout(
                analyzer, A, A_region, A_dims, M, K, A_dtype, "A register"
            )
        else:
            _require_full_region(analyzer, A, A_region, "A")
            _require_layout(A.layout, wgmma_a_register_layout(M, K, A_dtype), "A register")

    try:
        B_swizzle, _, B_sdo, B_mn_major = get_mma_smem_descriptor_params(
            B, B_region, B_dtype, trans_b
        )
        # PTX specifies this field as unused for all swizzled WGMMA
        # descriptors and requires the canonical encoded value one.
        B_ldo = 1
        if operand_mode == "ss":
            A_swizzle, _, A_sdo, A_mn_major = get_mma_smem_descriptor_params(
                A, A_region, A_dtype, trans_a
            )
            A_ldo = 1
    except ValueError as err:
        fail(f"WGMMA shared-memory descriptor mismatch: {err}")

    A_k_dim = A_dims[0][0] if trans_a else A_dims[1][0]
    B_k_dim = B_dims[0][0] if trans_b else B_dims[1][0]
    B_n_dim = B_dims[1][0] if trans_b else B_dims[0][0]
    K_tiles = K // _WGMMA_K
    N_tiles = N // instruction_n
    n_accums = M * N // 128
    n_accums_per_mma = M * instruction_n // 128
    accum_expr = tvm.tirx.const(int(accum), "bool")

    # fmt: off
    if operand_mode == "ss":
        @T.prim_func(check_well_formed=False)
        def impl():
            C_local = C.local()
            if not accum:
                for i in T.unroll(n_accums):
                    C_local[C_local_offset + i] = T.float32(0)
            for i in T.unroll(n_accums):
                T.ptx.wgmma.noop_barrier(C_local[C_local_offset + i])
            T.ptx.wgmma.fence()
            for ni in T.unroll(N_tiles):
                for ki in T.unroll(K_tiles):
                    scale_d = T.meta_var(tvm.tirx.any(ki != 0, accum_expr))
                    T.ptx.wgmma.mma_async.ss(
                        T.ptx.wgmma.make_matrix_descriptor(
                            A.ptr_to(_descriptor_starts(A_region, A_k_dim, ki)),
                            A_ldo,
                            A_sdo,
                            A_swizzle.value,
                        ),
                        T.ptx.wgmma.make_matrix_descriptor(
                            B.ptr_to(
                                _descriptor_starts(
                                    B_region,
                                    B_k_dim,
                                    ki,
                                    B_n_dim,
                                    ni,
                                    instruction_n,
                                )
                            ),
                            B_ldo,
                            B_sdo,
                            B_swizzle.value,
                        ),
                        *[
                            C_local[C_local_offset + ni * n_accums_per_mma + i]
                            for i in range(n_accums_per_mma)
                        ],
                        M=M,
                        N=instruction_n,
                        K=_WGMMA_K,
                        in_dtype=input_dtype,
                        out_dtype=C_dtype,
                        transA=A_mn_major,
                        transB=B_mn_major,
                        scaleA=1.0,
                        scaleB=1.0,
                        scaleD=scale_d,
                    )
    else:
        n_a_regs_per_mma = M * _WGMMA_K // 128 // 2

        if lhs_cast is None:
            n_a_regs = M * K // 128 // 2

            @T.prim_func(check_well_formed=False)
            def impl():
                C_local = C.local()
                A_u32 = A.view("uint32")
                A_local = A_u32.local(n_a_regs)
                if not accum:
                    for i in T.unroll(n_accums):
                        C_local[C_local_offset + i] = T.float32(0)
                for i in T.unroll(n_a_regs):
                    T.ptx.wgmma.noop_barrier(A_local[i])
                for i in T.unroll(n_accums):
                    T.ptx.wgmma.noop_barrier(C_local[C_local_offset + i])
                T.ptx.wgmma.fence()
                for ni in T.unroll(N_tiles):
                    for ki in T.unroll(K_tiles):
                        scale_d = T.meta_var(tvm.tirx.any(ki != 0, accum_expr))
                        T.ptx.wgmma.mma_async.rs(
                            T.ptx.wgmma.make_matrix_descriptor(
                                B.ptr_to(
                                    _descriptor_starts(
                                        B_region,
                                        B_k_dim,
                                        ki,
                                        B_n_dim,
                                        ni,
                                        instruction_n,
                                    )
                                ),
                                B_ldo,
                                B_sdo,
                                B_swizzle.value,
                            ),
                            *[
                                A_local[ki * n_a_regs_per_mma + i]
                                for i in range(n_a_regs_per_mma)
                            ],
                            *[
                                C_local[C_local_offset + ni * n_accums_per_mma + i]
                                for i in range(n_accums_per_mma)
                            ],
                            M=M,
                            N=instruction_n,
                            K=_WGMMA_K,
                            in_dtype=input_dtype,
                            out_dtype=C_dtype,
                            transA=False,
                            transB=B_mn_major,
                            scaleA=1.0,
                            scaleB=1.0,
                            scaleD=scale_d,
                        )
        else:
            n_a_values = M * K // 128
            n_a_regs = M * K // 128 // 2

            @T.prim_func(check_well_formed=False)
            def impl():
                C_local = C.local()
                A_local = A.local(n_a_values)
                A_packed = T.alloc_buffer((n_a_regs,), "uint32", scope="local")
                if not accum:
                    for i in T.unroll(n_accums):
                        C_local[C_local_offset + i] = T.float32(0)
                for i in T.unroll(n_a_regs):
                    A_packed[i] = T.cuda.float22bfloat162_rn(
                        A_local[i * 2], A_local[i * 2 + 1]
                    )
                    T.ptx.wgmma.noop_barrier(A_packed[i])
                for i in T.unroll(n_accums):
                    T.ptx.wgmma.noop_barrier(C_local[C_local_offset + i])
                T.ptx.wgmma.fence()
                for ni in T.unroll(N_tiles):
                    for ki in T.unroll(K_tiles):
                        scale_d = T.meta_var(tvm.tirx.any(ki != 0, accum_expr))
                        T.ptx.wgmma.mma_async.rs(
                            T.ptx.wgmma.make_matrix_descriptor(
                                B.ptr_to(
                                    _descriptor_starts(
                                        B_region,
                                        B_k_dim,
                                        ki,
                                        B_n_dim,
                                        ni,
                                        instruction_n,
                                    )
                                ),
                                B_ldo,
                                B_sdo,
                                B_swizzle.value,
                            ),
                            *[
                                A_packed[ki * n_a_regs_per_mma + i]
                                for i in range(n_a_regs_per_mma)
                            ],
                            *[
                                C_local[C_local_offset + ni * n_accums_per_mma + i]
                                for i in range(n_accums_per_mma)
                            ],
                            M=M,
                            N=instruction_n,
                            K=_WGMMA_K,
                            in_dtype=input_dtype,
                            out_dtype=C_dtype,
                            transA=False,
                            transB=B_mn_major,
                            scaleA=1.0,
                            scaleB=1.0,
                            scaleD=scale_d,
                        )
    # fmt: on
    return impl


@register_dispatch(
    "gemm_async",
    "cuda",
    variant="wgmma",
    priority=20,
    when=[
        predicate(
            "sm90a_only",
            cuda_arch_matches,
            min_version=90,
            max_version=100,
            require_suffix="a",
        ),
        predicate("full_warpgroup", _full_warpgroup),
    ],
)
def gemm_async_dispatch_wgmma(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Dispatch the accepted SM90a slice to WGMMA."""

    return gemm_async_wgmma_impl(op_call, sctx)
