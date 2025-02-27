/*
 * RISC-V translation routines for the P Standard Extensions.
 *
 * Copyright (c) 2020-2023 PLCT Lab
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#define GEN_SIMD_TRANS(NAME)                                \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
   TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);              \
   TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);              \
   TCGv dest = dest_gpr(ctx, a->rd);                        \
   gen_helper_##NAME(dest, tcg_env, src1, src2);            \
   return true;
}

#if defined(TARGET_RISCV32)                                 \
#define GEN_SIMD_TRANS_64(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
   REQUIRE_64BIT(ctx);                                      \
   return true;                                             \
}                                                           \
#else                                                       \
#define GEN_SIMD_TRANS_64(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, src2);           \
    return true;                                            \
}
#endif

GEN_SIMD_TRANS(padd_b)
GEN_SIMD_TRANS(padd_h)

