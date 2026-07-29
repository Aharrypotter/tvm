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

"""High-level ``Tx.gemm_async`` tests for the SM90a WGMMA dispatch."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
    mma_shared_layout,
    mma_shared_layout_k_major,
)
from tvm.tirx.layout import wgmma_a_register_layout, wgmma_accumulator_layout
from tvm.tirx.operator.tile_primitive import list_registered_schedules

pytestmark = pytest.mark.cuda_sm90


def _build_wgmma(
    mode,
    dtype="float16",
    n=64,
    k=64,
    accum=False,
    dispatch="wgmma",
    trans_a=False,
    instruction_n=None,
):
    m = 64
    effective_instruction_n = min(n, 64) if instruction_n is None else instruction_n
    a_smem_layout = mma_shared_layout(dtype, 3, (m, k))
    b_swizzle = 1 if n == 16 else 2 if n == 32 else 3
    b_smem_layout = (
        mma_shared_layout_k_major(dtype, b_swizzle, (k, n))
        if effective_instruction_n == 128
        else mma_shared_layout(dtype, b_swizzle, (k, n))
    )
    c_layout = wgmma_accumulator_layout(m, n)

    if mode == "ss":

        @T.prim_func
        def gemm(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, D_ptr: T.handle):
            A_g = T.match_buffer(A_ptr, (m, k), dtype)
            B_g = T.match_buffer(B_ptr, (k, n), dtype)
            C_g = T.match_buffer(C_ptr, (m, n), "float32")
            D_g = T.match_buffer(D_ptr, (m, n), "float32")
            T.device_entry()
            _cta = T.cta_id([1])
            _wg = T.warpgroup_id([1])
            warp = T.warp_id_in_wg([4])
            lane = T.lane_id([32])
            tid = warp * 32 + lane
            A_s = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_smem_layout)
            B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_smem_layout)
            C_f = T.alloc_buffer((m, n), "float32", scope="local", layout=c_layout)
            for i in T.serial((m * k + 127) // 128):
                linear = i * 128 + tid
                if linear < m * k:
                    A_s[linear // k, linear % k] = A_g[linear // k, linear % k]
            for i in T.serial((k * n + 127) // 128):
                linear = i * 128 + tid
                if linear < k * n:
                    B_s[linear // n, linear % n] = B_g[linear // n, linear % n]
            if accum:
                C_local = C_f.local(n // 2)
                for s in T.unroll(n // 2):
                    rN = s % 2
                    rM = s // 2 % 2
                    nt = s // 4
                    C_local[s] = C_g[
                        warp * 16 + lane // 4 + rM * 8,
                        nt * 8 + lane % 4 * 2 + rN,
                    ]
            T.cuda.cta_sync()
            Tx.warpgroup.gemm_async(
                C_f,
                A_s,
                B_s,
                transA=trans_a,
                transB=True,
                accum=accum,
                dispatch=dispatch,
                wgmma_instruction_n=effective_instruction_n,
            )
            T.ptx.wgmma.commit_group()
            T.ptx.wgmma.wait_group(0)
            C_local = C_f.local(n // 2)
            for s in T.unroll(n // 2):
                T.ptx.wgmma.noop_barrier(C_local[s])
            for s in T.unroll(n // 2):
                rN = s % 2
                rM = s // 2 % 2
                nt = s // 4
                D_g[
                    warp * 16 + lane // 4 + rM * 8,
                    nt * 8 + lane % 4 * 2 + rN,
                ] = C_local[s]

        return gemm

    if mode != "rs":
        raise ValueError(f"unknown mode {mode}")
    a_reg_layout = wgmma_a_register_layout(m, k, dtype)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (m, k), dtype)
        B_g = T.match_buffer(B_ptr, (k, n), dtype)
        C_g = T.match_buffer(C_ptr, (m, n), "float32")
        D_g = T.match_buffer(D_ptr, (m, n), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _wg = T.warpgroup_id([1])
        warp = T.warp_id_in_wg([4])
        lane = T.lane_id([32])
        tid = warp * 32 + lane
        A_f = T.alloc_buffer((m, k), dtype, scope="local", layout=a_reg_layout)
        B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_smem_layout)
        C_f = T.alloc_buffer((m, n), "float32", scope="local", layout=c_layout)
        A_local = A_f.local(m * k // 128)
        for s in T.unroll(m * k // 128):
            kp = s % 2
            rM = s // 2 % 2
            k8 = s // 4
            A_local[s] = A_g[
                warp * 16 + lane // 4 + rM * 8,
                k8 * 8 + lane % 4 * 2 + kp,
            ]
        for i in T.serial((k * n + 127) // 128):
            linear = i * 128 + tid
            if linear < k * n:
                B_s[linear // n, linear % n] = B_g[linear // n, linear % n]
        if accum:
            C_local = C_f.local(n // 2)
            for s in T.unroll(n // 2):
                rN = s % 2
                rM = s // 2 % 2
                nt = s // 4
                C_local[s] = C_g[
                    warp * 16 + lane // 4 + rM * 8,
                    nt * 8 + lane % 4 * 2 + rN,
                ]
        T.cuda.cta_sync()
        Tx.warpgroup.gemm_async(
            C_f,
            A_f,
            B_s,
            transA=trans_a,
            transB=True,
            accum=accum,
            dispatch=dispatch,
            wgmma_instruction_n=effective_instruction_n,
        )
        T.ptx.wgmma.commit_group()
        T.ptx.wgmma.wait_group(0)
        C_local = C_f.local(n // 2)
        for s in T.unroll(n // 2):
            T.ptx.wgmma.noop_barrier(C_local[s])
        for s in T.unroll(n // 2):
            rN = s % 2
            rM = s // 2 % 2
            nt = s // 4
            D_g[
                warp * 16 + lane // 4 + rM * 8,
                nt * 8 + lane % 4 * 2 + rN,
            ] = C_local[s]

    return gemm


def _lower(func, arch="sm_90a"):
    with tvm.target.Target({"kind": "cuda", "arch": arch}):
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))


def _build_wgmma_accumulator_lhs(n=64, k=128):
    """Build RS WGMMA that narrows an FP32 accumulator-layout lhs per k16 issue."""

    m = 64
    dtype = "bfloat16"
    a_layout = wgmma_accumulator_layout(m, k)
    b_swizzle = 1 if n == 16 else 2 if n == 32 else 3
    b_layout = mma_shared_layout(dtype, b_swizzle, (k, n))
    c_layout = wgmma_accumulator_layout(m, n)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (m, k), "float32")
        B_g = T.match_buffer(B_ptr, (k, n), dtype)
        D_g = T.match_buffer(D_ptr, (m, n), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _wg = T.warpgroup_id([1])
        warp = T.warp_id_in_wg([4])
        lane = T.lane_id([32])
        tid = warp * 32 + lane
        A_f = T.alloc_buffer((m, k), "float32", scope="local", layout=a_layout)
        B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_layout)
        C_f = T.alloc_buffer((m, n), "float32", scope="local", layout=c_layout)
        A_local = A_f.local(m * k // 128)
        for s in T.unroll(m * k // 128):
            kp = s % 2
            rM = s // 2 % 2
            k8 = s // 4
            A_local[s] = A_g[
                warp * 16 + lane // 4 + rM * 8,
                k8 * 8 + lane % 4 * 2 + kp,
            ]
        for i in T.serial((k * n + 127) // 128):
            linear = i * 128 + tid
            if linear < k * n:
                B_s[linear // n, linear % n] = B_g[linear // n, linear % n]
        T.cuda.cta_sync()
        Tx.warpgroup.gemm_async(
            C_f,
            A_f,
            B_s,
            transA=False,
            transB=True,
            accum=False,
            dispatch="wgmma",
            lhs_cast="bfloat16",
        )
        T.ptx.wgmma.commit_group()
        T.ptx.wgmma.wait_group(0)
        C_local = C_f.local(n // 2)
        for s in T.unroll(n // 2):
            T.ptx.wgmma.noop_barrier(C_local[s])
        for s in T.unroll(n // 2):
            rN = s % 2
            rM = s // 2 % 2
            nt = s // 4
            D_g[
                warp * 16 + lane // 4 + rM * 8,
                nt * 8 + lane % 4 * 2 + rN,
            ] = C_local[s]

    return gemm


def _build_wgmma_accumulator_output_slices():
    """Build two WGMMA calls into disjoint N64 views of one N128 accumulator."""

    m, n, k = 64, 128, 64
    dtype = "bfloat16"
    a_layout = mma_shared_layout(dtype, 3, (m, k))
    b_layout = mma_shared_layout(dtype, 3, (k, n))
    c_layout = wgmma_accumulator_layout(m, n)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (m, k), dtype)
        B_g = T.match_buffer(B_ptr, (k, n), dtype)
        D_g = T.match_buffer(D_ptr, (m, n), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _wg = T.warpgroup_id([1])
        warp = T.warp_id_in_wg([4])
        lane = T.lane_id([32])
        tid = warp * 32 + lane
        A_s = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_layout)
        B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_layout)
        C_f = T.alloc_buffer((m, n), "float32", scope="local", layout=c_layout)
        for i in T.serial((m * k + 127) // 128):
            linear = i * 128 + tid
            A_s[linear // k, linear % k] = A_g[linear // k, linear % k]
        for i in T.serial((k * n + 127) // 128):
            linear = i * 128 + tid
            B_s[linear // n, linear % n] = B_g[linear // n, linear % n]
        T.cuda.cta_sync()
        Tx.warpgroup.gemm_async(
            C_f[:, 0:64],
            A_s,
            B_s[:, 0:64],
            transA=False,
            transB=True,
            accum=False,
            dispatch="wgmma",
        )
        T.ptx.wgmma.commit_group()
        T.ptx.wgmma.wait_group(0)
        Tx.warpgroup.gemm_async(
            C_f[:, 64:128],
            A_s,
            B_s[:, 64:128],
            transA=False,
            transB=True,
            accum=False,
            dispatch="wgmma",
        )
        T.ptx.wgmma.commit_group()
        T.ptx.wgmma.wait_group(0)
        C_local = C_f.local(n // 2)
        for s in T.unroll(n // 2):
            T.ptx.wgmma.noop_barrier(C_local[s])
        for s in T.unroll(n // 2):
            rN = s % 2
            rM = s // 2 % 2
            nt = s // 4
            D_g[
                warp * 16 + lane // 4 + rM * 8,
                nt * 8 + lane % 4 * 2 + rN,
            ] = C_local[s]

    return gemm


def _build_staged_wgmma_n128():
    """Build native N128 from one slice of a staged K-major B buffer."""

    m, n, k, stages = 64, 128, 64, 2
    dtype = "bfloat16"
    a_layout = mma_shared_layout(dtype, 3, (m, k))
    b_layout = mma_shared_layout_k_major(dtype, 3, (stages, k, n))
    c_layout = wgmma_accumulator_layout(m, n)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle):
        T.device_entry()
        _cta = T.cta_id([1])
        _wg = T.warpgroup_id([1])
        _warp = T.warp_id_in_wg([4])
        _lane = T.lane_id([32])
        A_s = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_layout)
        B_s = T.alloc_buffer((stages, k, n), dtype, scope="shared", layout=b_layout)
        C_f = T.alloc_buffer((m, n), "float32", scope="local", layout=c_layout)
        Tx.warpgroup.gemm_async(
            C_f,
            A_s,
            B_s[1, :, :],
            transA=False,
            transB=True,
            accum=False,
            dispatch="wgmma",
            wgmma_instruction_n=128,
        )

    return gemm


def _build_invalid_wgmma(case):
    m, n, k = 64, 64, 64
    c_layout = wgmma_accumulator_layout(m, n)

    if case == "dtype":
        dtype = "float32"
        a_layout = mma_shared_layout(dtype, 3, (m, k))
        b_layout = mma_shared_layout(dtype, 3, (k, n))

        @T.prim_func
        def invalid(A_ptr: T.handle, B_ptr: T.handle):
            T.device_entry()
            _cta = T.cta_id([1])
            _wg = T.warpgroup_id([1])
            _warp = T.warp_id_in_wg([4])
            _lane = T.lane_id([32])
            A_s = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_layout)
            B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_layout)
            C_f = T.alloc_buffer((m, n), "float32", scope="local", layout=c_layout)
            Tx.warpgroup.gemm_async(C_f, A_s, B_s, transA=False, transB=True, dispatch="wgmma")

        return invalid

    dtype = "float16"
    a_layout = mma_shared_layout(dtype, 3, (m, k))
    b_layout = mma_shared_layout(dtype, 3, (k, n))

    if case == "c_scope":

        @T.prim_func
        def invalid(A_ptr: T.handle, B_ptr: T.handle):
            T.device_entry()
            _cta = T.cta_id([1])
            _wg = T.warpgroup_id([1])
            _warp = T.warp_id_in_wg([4])
            _lane = T.lane_id([32])
            A_s = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_layout)
            B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_layout)
            C_s = T.alloc_buffer((m, n), "float32", scope="shared")
            Tx.warpgroup.gemm_async(C_s, A_s, B_s, transA=False, transB=True, dispatch="wgmma")

        return invalid

    if case == "c_layout":

        @T.prim_func
        def invalid(A_ptr: T.handle, B_ptr: T.handle):
            T.device_entry()
            _cta = T.cta_id([1])
            _wg = T.warpgroup_id([1])
            _warp = T.warp_id_in_wg([4])
            _lane = T.lane_id([32])
            A_s = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_layout)
            B_s = T.alloc_buffer((k, n), dtype, scope="shared", layout=b_layout)
            C_f = T.alloc_buffer((m, n), "float32", scope="local")
            Tx.warpgroup.gemm_async(C_f, A_s, B_s, transA=False, transB=True, dispatch="wgmma")

        return invalid

    raise ValueError(f"unknown invalid case {case}")


def test_wgmma_variant_is_registered():
    schedules = list_registered_schedules()
    cuda_gemm_async = schedules.get("tirx.tile.gemm_async", {}).get("cuda", [])
    assert "wgmma" in cuda_gemm_async
    assert "tcgen05" in cuda_gemm_async
    assert cuda_gemm_async.index("wgmma") < cuda_gemm_async.index("tcgen05")


@pytest.mark.parametrize("dtype", ["float32", "uint32"])
def test_wgmma_noop_barrier_codegen_accepts_buffer_load(dtype):
    from tvm.backend.cuda.operator.intrinsics.wgmma import (
        codegen_ptx_wgmma_noop_barrier,
    )

    buffer = tvm.tirx.decl_buffer((1,), dtype)
    assert codegen_ptx_wgmma_noop_barrier([buffer[0]])


def test_wgmma_is_selected_automatically_on_sm90a():
    script = _lower(_build_wgmma("ss", dispatch=None))["main"].script()
    assert "T.ptx.wgmma.mma_async.ss" in script
    assert "T.ptx.tcgen05" not in script


@pytest.mark.parametrize("mode", ["ss", "rs"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("n", [16, 32, 64, 128])
def test_wgmma_lowers_from_high_level_gemm_async(mode, dtype, n):
    script = _lower(_build_wgmma(mode, dtype=dtype, n=n))["main"].script()
    assert f"T.ptx.wgmma.mma_async.{mode}" in script
    expected_instruction_n = min(n, 64)
    assert f"T.ptx.wgmma.mma_async.{mode}(64, {expected_instruction_n}, 16" in script
    assert f"T.ptx.wgmma.mma_async.{mode}(64, 128, 16" not in script
    assert "T.ptx.wgmma.make_matrix_descriptor" in script
    expected_sdo, expected_swizzle = (16, 1) if n == 16 else (32, 2) if n == 32 else (64, 3)
    assert f", 1, {expected_sdo}, {expected_swizzle})" in script
    assert "T.ptx.wgmma.fence()" in script
    assert "T.ptx.wgmma.commit_group()" in script
    assert "T.ptx.wgmma.wait_group(0)" in script
    assert "T.ptx.tcgen05" not in script


@pytest.mark.parametrize("mode", ["ss", "rs"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_wgmma_native_n128_lowers_from_high_level_gemm_async(mode, dtype):
    script = _lower(_build_wgmma(mode, dtype=dtype, n=128, instruction_n=128))["main"].script()
    assert f"T.ptx.wgmma.mma_async.{mode}(64, 128, 16" in script
    assert f"T.ptx.wgmma.mma_async.{mode}(64, 64, 16" not in script
    assert "T.ptx.tcgen05" not in script


def test_wgmma_native_n128_accepts_staged_k_major_b_layout():
    script = _lower(_build_staged_wgmma_n128())["main"].script()
    assert "T.ptx.wgmma.mma_async.ss(64, 128, 16" in script
    assert "T.ptx.wgmma.mma_async.ss(64, 64, 16" not in script


@pytest.mark.parametrize(
    "n,instruction_n,error",
    [
        (128, 24, "wgmma_instruction_n must be one of"),
        (32, 64, "logical N=32 must be divisible"),
        (64, 128, "logical N=64 must be divisible"),
    ],
)
def test_wgmma_rejects_invalid_instruction_n(n, instruction_n, error):
    with pytest.raises(RuntimeError, match=error):
        _lower(_build_wgmma("ss", n=n, instruction_n=instruction_n))


def test_wgmma_rs_can_narrow_accumulator_lhs_per_instruction():
    script = _lower(_build_wgmma_accumulator_lhs())["main"].script()
    assert "for ki in T.unroll(8):" in script
    assert script.count("T.ptx.wgmma.mma_async.rs(64, 64, 16") == 1
    assert script.count("T.cuda.float22bfloat162_rn") == 1
    assert script.count("T.ptx.wgmma.fence()") == 1
    pack_pos = script.index("T.cuda.float22bfloat162_rn")
    fence_pos = script.index("T.ptx.wgmma.fence()")
    issue_loop_pos = script.index("for ki in T.unroll(8):")
    assert pack_pos < fence_pos < issue_loop_pos
    assert "A_packed" in script
    assert "T.ptx.tcgen05" not in script


@pytest.mark.parametrize("n", [16, 32])
def test_wgmma_narrow_accumulator_lhs_prepacked_before_single_fence(n):
    script = _lower(_build_wgmma_accumulator_lhs(n=n, k=64))["main"].script()
    assert "for ki in T.unroll(4):" in script
    assert script.count(f"T.ptx.wgmma.mma_async.rs(64, {n}, 16") == 1
    assert script.count("T.ptx.wgmma.fence()") == 1
    pack_pos = script.index("T.cuda.float22bfloat162_rn")
    fence_pos = script.index("T.ptx.wgmma.fence()")
    issue_loop_pos = script.index("for ki in T.unroll(4):")
    assert pack_pos < fence_pos < issue_loop_pos
    assert "A_packed" in script


def test_wgmma_preserves_accumulator_output_slice_offsets():
    script = _lower(_build_wgmma_accumulator_output_slices())["main"].script()
    mma_lines = [line.strip() for line in script.splitlines() if "mma_async.ss" in line]
    assert len(mma_lines) == 2
    assert "C_local[0]" in mma_lines[0]
    assert "C_local_1[2]" in mma_lines[1]
    assert "C_local_1[0]" not in mma_lines[1]


@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_100a"])
def test_forced_wgmma_rejects_wrong_architecture(arch):
    with pytest.raises(RuntimeError, match="sm90a_only"):
        _lower(_build_wgmma("ss"), arch=arch)


@pytest.mark.parametrize(
    "case, error",
    [
        ("dtype", "requires matching float16/bfloat16 inputs"),
        ("c_scope", "requires C in register"),
        ("c_layout", "C accumulator fragment layout mismatch"),
    ],
)
def test_forced_wgmma_rejects_invalid_contract(case, error):
    with pytest.raises(RuntimeError, match=error):
        _lower(_build_invalid_wgmma(case))


def test_wgmma_rs_rejects_transposed_register_a():
    with pytest.raises(RuntimeError, match="RS requires transA=False"):
        _lower(_build_wgmma("rs", trans_a=True))


@pytest.mark.parametrize("mode", ["ss", "rs"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("accum", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9, exact=True), reason="need cuda compute == 9.0")
def test_wgmma_numerical(mode, dtype, n, accum):
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = ml_dtypes.bfloat16
    else:
        np_dtype = np.float16

    func = _build_wgmma(mode, dtype=dtype, n=n, accum=accum)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert "wgmma.mma_async" in source
    assert "tcgen05" not in source

    rng = np.random.default_rng(0)
    A_np = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
    B_np = rng.uniform(-1, 1, (64, n)).astype(np.float32)
    C_np = rng.uniform(-1, 1, (64, n)).astype(np.float32)
    A_input = A_np.astype(np_dtype)
    B_input = B_np.astype(np_dtype)
    expected = A_input.astype(np.float32) @ B_input.astype(np.float32)
    expected += C_np if accum else 0

    def run_and_check():
        dev = tvm.cuda(0)
        A_dev = tvm.runtime.tensor(A_input, dev)
        B_dev = tvm.runtime.tensor(B_input, dev)
        C_dev = tvm.runtime.tensor(C_np, dev)
        outputs = [tvm.runtime.tensor(np.zeros((64, n), np.float32), dev) for _ in range(3)]
        mod(A_dev, B_dev, C_dev, outputs[0])
        mod(A_dev, B_dev, C_dev, outputs[1])
        dev.sync()

        stream = dev.create_raw_stream()
        try:
            dev.set_raw_stream(stream)
            mod(A_dev, B_dev, C_dev, outputs[2])
            dev.sync(stream)
        finally:
            dev.set_raw_stream(0)
            dev.free_raw_stream(stream)

        actual = [out.numpy() for out in outputs]
        for result in actual:
            tvm.testing.assert_allclose(expected, result, atol=3e-2, rtol=3e-2)
        np.testing.assert_array_equal(actual[0], actual[1])
        np.testing.assert_array_equal(actual[0], actual[2])

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("mode", ["ss", "rs"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("accum", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9, exact=True), reason="need cuda compute == 9.0")
def test_wgmma_native_n128_numerical(mode, dtype, accum):
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = ml_dtypes.bfloat16
    else:
        np_dtype = np.float16

    func = _build_wgmma(mode, dtype=dtype, n=128, accum=accum, instruction_n=128)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert source.count("wgmma.mma_async") == 1
    assert "wgmma.mma_async.sync.aligned.m64n128k16" in source
    assert "wgmma.mma_async.sync.aligned.m64n64k16" not in source
    assert "tcgen05" not in source

    rng = np.random.default_rng(0)
    A_np = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
    B_np = rng.uniform(-1, 1, (64, 128)).astype(np.float32)
    C_np = rng.uniform(-1, 1, (64, 128)).astype(np.float32)
    A_input = A_np.astype(np_dtype)
    B_input = B_np.astype(np_dtype)
    expected = A_input.astype(np.float32) @ B_input.astype(np.float32)
    expected += C_np if accum else 0

    def run_and_check():
        dev = tvm.cuda(0)
        A_dev = tvm.runtime.tensor(A_input, dev)
        B_dev = tvm.runtime.tensor(B_input, dev)
        C_dev = tvm.runtime.tensor(C_np, dev)
        D_dev = tvm.runtime.tensor(np.zeros((64, 128), np.float32), dev)
        mod(A_dev, B_dev, C_dev, D_dev)
        dev.sync()
        tvm.testing.assert_allclose(expected, D_dev.numpy(), atol=3e-2, rtol=3e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9, exact=True), reason="need cuda compute == 9.0")
@pytest.mark.parametrize("n,k", [(64, 128), (32, 64), (16, 128)])
def test_wgmma_accumulator_lhs_numerical(n, k):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    func = _build_wgmma_accumulator_lhs(n=n, k=k)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert source.count("wgmma.mma_async") == 1
    assert source.count("tvm_builtin_float22bfloat162_rn") >= 1
    assert "A_packed" in source
    pack_pos = source.index("A_packed")
    fence_pos = source.index("ptx_wgmma_fence();", pack_pos)
    mma_pos = source.index("ptx_wgmma_mma_async_rs", fence_pos)
    assert pack_pos < fence_pos < mma_pos
    assert "descB_ptr" not in source
    assert "tcgen05" not in source

    rng = np.random.default_rng(0)
    A_input = rng.uniform(-1, 1, (64, k)).astype(np.float32)
    B_input = rng.uniform(-1, 1, (k, n)).astype(ml_dtypes.bfloat16)
    expected = A_input.astype(ml_dtypes.bfloat16).astype(np.float32) @ B_input.astype(np.float32)
    dev = tvm.cuda(0)
    A_dev = tvm.runtime.tensor(A_input, dev)
    B_dev = tvm.runtime.tensor(B_input, dev)
    D_dev = tvm.runtime.tensor(np.zeros((64, n), np.float32), dev)
    mod(A_dev, B_dev, D_dev)
    dev.sync()
    tvm.testing.assert_allclose(expected, D_dev.numpy(), atol=3e-2, rtol=3e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9, exact=True), reason="need cuda compute == 9.0")
def test_wgmma_accumulator_output_slices_numerical():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    func = _build_wgmma_accumulator_output_slices()
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert "wgmma.mma_async.sync.aligned.m64n64k16" in source
    assert source.count("ptx_wgmma_commit_group();") == 2
    assert "tcgen05" not in source

    rng = np.random.default_rng(0)
    A_input = rng.uniform(-1, 1, (64, 64)).astype(ml_dtypes.bfloat16)
    B_input = rng.uniform(-1, 1, (64, 128)).astype(ml_dtypes.bfloat16)
    expected = A_input.astype(np.float32) @ B_input.astype(np.float32)
    dev = tvm.cuda(0)
    A_dev = tvm.runtime.tensor(A_input, dev)
    B_dev = tvm.runtime.tensor(B_input, dev)
    D_dev = tvm.runtime.tensor(np.zeros((64, 128), np.float32), dev)
    mod(A_dev, B_dev, D_dev)
    dev.sync()
    tvm.testing.assert_allclose(expected, D_dev.numpy(), atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    tvm.testing.main()
