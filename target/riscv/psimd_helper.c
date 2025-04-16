/*
 * RISC-V Packed SIMD Extension Helpers for QEMU.
 *
 * Copyright (C) 2024 PLCT Lab.
 * Written by Codethink Ltd and SiFive.
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

#include "qemu/osdep.h"
#include "cpu.h"
#include "qemu/host-utils.h"
#include "exec/exec-all.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include "internals.h"

static int64_t signed_saturate(CPURISCVState *env, int64_t arg, int range)
{
    int64_t max = (1ull << (range - 1)) - 1;
    int64_t min = -(1 << (range - 1));

    if (arg > max) {
        arg = max;
        env->vxsat = 0x1;
    } else if (arg < min) {
        arg = min;
        env->vxsat = 0x1;
    }

    return arg;
}

/*
static int64_t signed_saturate64(CPURISCVState *env, Int128 arg)
{
    Int128 max = INT64_MAX;
    Int128 min = INT64_MIN;

    if (arg > max) {
        arg = max;
        env->vxsat = 0x1;
    } else if (arg < min) {
        arg = min;
        env->vxsat = 0x1;
    }
                                                                                                                                                                                                      return arg;
}
*/

static uint64_t unsigned_saturate(CPURISCVState *env, uint64_t arg, int range)
{
    uint64_t max = (1ull << range) - 1;

    if (arg > max) {
        arg = max;
        env->vxsat = 0x1;
    }
                                                                                                                                                                                                      return arg;
}

/*
static uint64_t unsigned_saturate64(CPURISCVState *env, __uint128_t arg)
{
    __uint128_t max = UINT64_MAX; 

    if (arg > max){
        arg = max;
        env->vxsat = 0x1;
    }

    return arg;
}
*/

target_ulong HELPER(padd_h)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_hs)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_b)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_bs)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(psub_b)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(psub_h)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(psadd_b)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i =0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(psadd_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(psaddu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)unsigned_saturate(env, v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(psaddu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(pssub_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(pssub_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(pssubu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)unsigned_saturate(env, v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(pssubu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(paadd_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paadd_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paaddu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paaddu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasub_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasub_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasubu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasubu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(psabs_b)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];

        if (v1 == INT8_MIN) {
            v1 = INT8_MAX;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = -v1;
        }

        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(pdifsumu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    target_long t = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = (target_long)(v1 - v2);
        t = t > 0 ? t : -t;
        rd = rd + t;
    }

    return rd;
}

target_ulong HELPER(pdifsumau_b)(CPURISCVState *env, target_ulong rs1,
                            target_ulong rs2, target_ulong rd)
{
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    target_long t = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = (target_long)(v1 -v2);
        t = t > 0 ? t : -t;
        rd = rd + t;
    }

    return rd;
}

target_ulong HELPER(pas_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psa_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psas_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(pssa_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(paas_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(pasa_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psabs_h)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];

        if (v1 == INT16_MIN) {
            v1 = INT16_MAX;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = -v1;
        }

        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(psh1add_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)((v1 << 1) + v2);
    }

    return rd;
}

target_ulong HELPER(pssh1sadd_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p  = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
	if ( v1 < 0xC000 ){
	    v1 = 0x8000;
            env->vxsat = 0x1;
	}else if ( v1 >= 0x4000){
	    v1 = 0x7FFF;
	    env->vxsat = 0x1;
	}else{
	    v1 = v1 < 1;
	}
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(pdif_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(pdifu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(pdif_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(pdifu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(predsum_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_long rd = rs2;
    int16_t *rs1_p = (int16_t *)&rs1;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (target_ulong)rd;
}

target_ulong HELPER(predsumu_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = rs2;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

target_ulong HELPER(predsum_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_long rd = rs2;
    int8_t *rs1_p = (int8_t *)&rs1;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (target_ulong)rd;
}

target_ulong HELPER(predsumu_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = rs2;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

target_ulong HELPER(sadd)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) + (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

target_ulong HELPER(saddu)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) + (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

target_ulong HELPER(ssub)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) - (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

target_ulong HELPER(ssubu)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) - (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

target_ulong HELPER(aadd)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

target_ulong HELPER(aaddu)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

target_ulong HELPER(asub)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];
    
    return (v1 - v2) >> 1;
}

target_ulong HELPER(asubu)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 - v2) >> 1;
}

target_ulong HELPER(ssh1sadd)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    target_long v1 = rs1_p[0];
    target_long v2 = rs2_p[0];

	if ( v1 < 0xC0000000 ){
	    v1 = 0x80000000;
        env->vxsat = 0x1;
	}else if ( v1 >= 0x40000000){
	    v1 = 0x7FFFFFFF;
	    env->vxsat = 0x1;
	}else{
	    v1 = v1 < 1;
	}
    
    return (int32_t)signed_saturate(env, v1 + v2, 32);
}

target_ulong HELPER(padd_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE / 4; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
    }

    return rd;:
}

target_ulong HELPER(psub_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(psadd_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
    }

    return rd;
}

target_ulong HELPER(psaddu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
    }

    return rd;
}

target_ulong HELPER(pssub_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
    }

    return rd;
}

target_ulong HELPER(pssubu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
    }

    return rd;
}

target_ulong HELPER(paadd_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paaddu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasub_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasubu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(psh1add_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 << 1) + v2);
    }

    return rd;
}

target_ulong HELPER(pssh1sadd_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p  = (int32_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
	if ( v1 < 0xC0000000 ){
	    v1 = 0x80000000;
            env->vxsat = 0x1;
	}else if ( v1 >= 0x40000000){
	    v1 = 0x7FFFFFFF;
	    env->vxsat = 0x1;
	}else{
	    v1 = v1 < 1;
	}
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 32);
    }

    return rd;
}

target_ulong HELPER(pas_wx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psa_wx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psas_wx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 - v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(pssa_wx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 + v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(paas_wx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)((v1 + v2) >> 1); 
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(pasa_wx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)((v1 - v2) >> 1); 
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(predsum_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_long rd = rs2;
    int32_t *rs1_p = (int32_t *)&rs1;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (target_ulong)rd;
}

target_ulong HELPER(predsumu_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = rs2;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

target_ulong HELPER(pslli_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 << imm);
    }

    return rd;
}

target_ulong HELPER(psll_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 << imm);
    }

    return rd;   
}

target_ulong HELPER(psrli_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint8_t)(v1 >> imm);
    }

    return rd;
}

target_ulong HELPER(psrl_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint8_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psrai_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 >> imm);
    }

    return rd;
}

target_ulong HELPER(psra_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 >> imm);
    }

    return rd;    
}

target_ulong HELPER(pmin_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pminu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;   
}

target_ulong HELPER(pmax_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pmaxu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmseq_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 == v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(pmslt_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFF : 0;
    }

    return rd;   
}

target_ulong HELPER(pmsltu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(psext_h_b)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 16, 8);
        t2 = sextract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(psati_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;     
}

target_ulong HELPER(pusati_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << immm) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(pslli_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 << imm);
    }

    return rd;     
}

target_ulong HELPER(psll_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 << imm);
    }

    return rd;      
}

target_ulong HELPER(psrli_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint16_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psrl_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint16_t)(v1 >> imm);
    }

    return rd;      
}

target_ulong HELPER(psrai_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psra_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psslai_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 << imm, 16);
    }

    return rd;     
}

target_ulong HELPER(psrari_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    if (imm == 0) {
        rd = rs1;
    } else {
        for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (imm - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(pssha_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        if ( imm <= -16){
            rd_p[i] = (int16_t) ( v1 >> 16 );
        }else if( imm > -16 && imm < 0) {
            rd_p[i] = (int16_t) ( v1 >> (-imm) );
        }else if ( imm >= 0 && imm < 16){
            rd_p[i] = (int16_t)signed_saturate(env, v1 << imm, 16);
        }else{
            rd_p[i] = (int16_t) 0x0000;
        }
    }

    return rd;
}

target_ulong HELPER(psshar_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
  
}

target_ulong HELPER(pmin_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;    
}

target_ulong HELPER(pminu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmax_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmaxu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pmseq_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 == v2 ? 0xFFFF : 0;
    }

    return rd;     
}

target_ulong HELPER(pmslt_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFFFF : 0;
    }

    return rd;    
}

target_ulong HELPER(pmsltu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFFFF : 0;
    }

    return rd;   
}

target_ulong HELPER(sati_32)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd; 
}

target_ulong HELPER(usati_32)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << shamt) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;     
}

target_ulong HELPER(sslai)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int64_t v1 = rs1_p[0];

    rd = signed_saturate(env, v1 << shamt, 32);

    return rd;     
}

target_ulong HELPER(srari_32)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    target_long v1 = rs1;

    if (shamt == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (shamt - 1)) + 1) >> 1;
    }

    return rd;     
}

target_ulong HELPER(ssha)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    int64_t xrs1 = (int64_t)rs1;

    if ( imm <= -32){
        rd = (int32_t) ( xrs1 >> 32 );
    }else if( imm > -32 && imm < 0) {
        rd = (int32_t) ( xrs1 >> (-imm) );
    }else if ( imm >= 0 && imm < 16){
        rd = (int32_t)signed_saturate(env, xrs1 << imm, 32);
    }else{
        rd = (int32_t) 0x00000000;
    }

    return rd;     
}

//saturating SHA with rounding
target_ulong HELPER(sshar)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    int32_t rd = 0;
    int64_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    int64_t xrs1 = (int64_t)rs1;

    if ( imm <= -32){
        xrd = (int64_t) ( xrs1 >> 32 );
        rd = (int32_t) (( xrd + 1 ) >> 1);
    }else if( imm > -32 && imm < 0) {
        xrd = (int64_t) ( xrs1 >> (-imm-1) );
        rd = (int32_t) (( xrd + 1 ) >> 1);
    }else if ( imm >= 0 && imm < 16){
        rd = (int32_t)signed_saturate(env, xrs1 << imm, 32);
    }else{
        rd = (int32_t) 0x00000000;
    }

    return (target_ulong)rd;
}

target_ulong HELPER(mseq)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    return rs1 == rs2 ? 0xFFFFFFFF : 0x00000000;
}

target_ulong HELPER(mslt)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    return (int32_t)rs1 < (int32_t)rs2 ? 0xFFFFFFFF : 0x00000000;
}

target_ulong HELPER(msltu)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    return (uint32_t)rs1 < (uint32_t)rs2 ? 0xFFFFFFFF : 0x00000000;
}

target_ulong HELPER(psext_w_b)(CPURISCVState *env, target_ulong rs1)
{
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    t1 = sextract32(&rs1, 32, 8);
    t2 = sextract32(&rs1, 0, 8);

    return (target_ulong)((t1 << 32) | (t2 & 0xFFFFFFFF));
}

target_ulong HELPER(psext_w_h)(CPURISCVState *env, target_ulong rs1)
{
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    t1 = sextract32(&rs1, 32, 16);
    t2 = sextract32(&rs1, 0, 16);

    return (target_ulong)((t1 << 32) | (t2 & 0xFFFFFFFF));
}

target_ulong HELPER(psati_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;
}

target_ulong HELPER(pusati_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << imm) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;     
}

target_ulong HELPER(pslli_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 << imm);
    }

    return rd;  
}

target_ulong HELPER(psll_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    uint64_t imm = rs2 & 0x1F;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 << imm);
    }

    return rd;     
}

target_ulong HELPER(psrli_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint32_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psrl_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t imm = rs2 & 0x1F;
    uint64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint32_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psrai_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psra_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    uint64_t imm = rs2 & 0x1F;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psslai_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 << shamt, 32);
    }

    return rd;     
}

target_ulong HELPER(psrari_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    if (shamt == 0) {
        rd = rs1;
    } else {
        for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;     
}

target_ulong HELPER(pssha_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        if ( imm <= -32){
            rd_p[i] = (int32_t) ( v1 >> 32 );
        }else if( imm > -32 && imm < 0) {
            rd_p[i] = (int32_t) ( v1 >> (-imm) );
        }else if ( imm >= 0 && imm < 32){
            rd_p[i] = (int32_t)signed_saturate(env, v1 << imm, 32);
        }else{
            rd_p[i] = (int32_t) 0x0000;
        }
    }

    return rd;     
}

target_ulong HELPER(psshar_ws)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++){
        int64_t xrs1 = (int64_t)rs1_p[i];
        if ( imm <= -32){
            xrd = (int64_t) ( xrs1 >> 32 );
            rd_p[i] = (int32_t) (( xrd + 1 ) >> 1);
        }else if( imm > -32 && imm < 0) {
            xrd = (int64_t) ( xrs1 >> (-imm-1) );
            rd_p[i] = (int32_t) (( xrd + 1 ) >> 1);
        }else if ( imm >= 0 && imm < 16){
            rd_p[i] = (int32_t)signed_saturate(env, xrs1 << imm, 32);
        }else{
            rd_p[i] = (int32_t) 0x00000000;
        }
    }

    return rd;
}

target_ulong HELPER(pmin_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pminu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmax_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pmaxu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmseq_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (v1 == v2) ? 0xFFFFFFFF : 0x00000000;
    }

    return rd;
}

target_ulong HELPER(pmslt_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (v1 < v2) ? 0xFFFFFFFF : 0x00000000;
    }

    return rd;     
}

target_ulong HELPER(pmsltu_w)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (v1 < v2) ? 0xFFFFFFFF : 0x00000000;
    }

    return rd;      
}

target_ulong HELPER(sati_64)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;
}

target_ulong HELPER(usati_64)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << imm) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;  
}

target_ulong HELPER(srari_64)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    target_long v1 = rs1;

    if (imm == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (imm - 1)) + 1) >> 1;
    }

    return rd;  
}

target_ulong HELPER(sha)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    __int128_t xrs1 = (__int128_t)rs1;

    if ( imm <= -64){
        rd = (int64_t) ( xrs1 >> 64 );
    }else if( imm > -64 && imm < 0) {
        rd = (int64_t) ( xrs1 >> (-imm) );
    }else{
        rd = rs1 << imm;
    }

    return rd;
}

target_ulong HELPER(shar)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    int64_t rd = 0;
    __int128_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    __int128_t xrs1 = (__int128_t)rs1;

    if ( imm <= -64){
        xrd = (__int128_t) ( xrs1 >> 64 );
        rd = (int64_t) (( xrd + 1 ) >> 1);
    }else if( imm > -64 && imm < 0) {
        xrd = (__int128_t) ( xrs1 >> (-imm-1) );
        rd = (int64_t) (( xrd + 1 ) >> 1);
    }else{
        rd = rs1 << imm;
    }

    return (target_ulong)rd;    
}