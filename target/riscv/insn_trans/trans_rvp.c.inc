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
   return true;                                             \
}                                                              

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_64(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
   REQUIRE_64BIT(ctx);                                      \
   return true;                                             \
}                                                           
#else
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

#define GEN_SIMD_TRANS_RD(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_RD_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_RD_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}
#endif

#define GEN_SIMD_TRANS_R1(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1);                 \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_R1_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_R1_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1);                 \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}
#endif

GEN_SIMD_TRANS(padd_b)
GEN_SIMD_TRANS(padd_bs)
GEN_SIMD_TRANS(psub_b)
GEN_SIMD_TRANS(psadd_b)
GEN_SIMD_TRANS(psaddu_b)
GEN_SIMD_TRANS(pssub_b)
GEN_SIMD_TRANS(pssubu_b)
GEN_SIMD_TRANS(paadd_b)
GEN_SIMD_TRANS(paaddu_b)
GEN_SIMD_TRANS(pasub_b)
GEN_SIMD_TRANS(pasubu_b)
GEN_SIMD_TRANS(pdifsumu_b)
GEN_SIMD_TRANS_RD(pdifsumau_b)
GEN_SIMD_TRANS_R1(psabs_b)
GEN_SIMD_TRANS(pdif_h)
GEN_SIMD_TRANS(pdifu_h)
GEN_SIMD_TRANS(predsum_bs)
GEN_SIMD_TRANS(predsumu_bs)

GEN_SIMD_TRANS(padd_h)
GEN_SIMD_TRANS(padd_hs)
GEN_SIMD_TRANS(psub_h)
GEN_SIMD_TRANS(psadd_h)
GEN_SIMD_TRANS(psaddu_h)
GEN_SIMD_TRANS(pssub_h)
GEN_SIMD_TRANS(pssubu_h)
GEN_SIMD_TRANS(paadd_h)
GEN_SIMD_TRANS(paaddu_h)
GEN_SIMD_TRANS(pasub_h)
GEN_SIMD_TRANS(pasubu_h)

GEN_SIMD_TRANS(psh1add_h)
GEN_SIMD_TRANS(pssh1sadd_h)

GEN_SIMD_TRANS(pas_hx)
GEN_SIMD_TRANS(psa_hx)
GEN_SIMD_TRANS(psas_hx)
GEN_SIMD_TRANS(pssa_hx)
GEN_SIMD_TRANS(paas_hx)
GEN_SIMD_TRANS(pasa_hx)

GEN_SIMD_TRANS(pdif_h)
GEN_SIMD_TRANS(pdifu_h)
GEN_SIMD_TRANS_R1(psabs_h)

GEN_SIMD_TRANS(predsum_hs)
GEN_SIMD_TRANS(predsumu_hs)

static bool trans_pli_b(DisasContext *ctx, arg_pli_b *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE){
        imm = (( imm << 8 ) + a->imm);
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_pli_h(DisasContext *ctx, arg_pli_b *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE / 2){
        imm =( imm << 16 ) + (a->imm & 0xFFFF);
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_plui_h(DisasContext *ctx, arg_pli_b *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE / 2){
        imm =( imm << 16 ) + (a->imm & 0xFFFF);
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}