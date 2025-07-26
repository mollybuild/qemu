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

uint32_t HELPER(sadd)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) + (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

uint32_t HELPER(saddu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) + (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

uint32_t HELPER(ssub)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) - (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

uint32_t HELPER(ssubu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) - (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

uint32_t HELPER(aadd)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

uint32_t HELPER(aaddu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

uint32_t HELPER(asub)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];
    
    return (v1 - v2) >> 1;
}

uint32_t HELPER(asubu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 - v2) >> 1;
}

uint32_t HELPER(ssh1sadd)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
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

uint64_t HELPER(padd_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE / 4; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
    }

    return rd;
}

uint64_t HELPER(padd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

    return rd;
}

uint64_t HELPER(psub_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(psadd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(psaddu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pssub_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pssubu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(paadd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(paaddu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pasub_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pasubu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(psh1add_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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
        rd_p[i] = (uint32_t)((v1 << 1) + v2);
    }

    return rd;
}

uint64_t HELPER(pssh1sadd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p  = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

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

uint64_t HELPER(pas_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(psa_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(psas_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pssa_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(paas_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pasa_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(predsum_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = rs2;
    int32_t *rs1_p = (int32_t *)&rs1;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (uint64_t)rd;
}

uint64_t HELPER(predsumu_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = rs2;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint64_t v1 = 0;

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
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    int32_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++){
        int32_t xrs1 = (int32_t)rs1_p[i];
        if ( imm <= -16){
            xrd = (int32_t) ( xrs1 >> 16 );
            rd_p[i] = (int16_t) (( xrd + 1 ) >> 1);
        }else if( imm > -16 && imm < 0) {
            xrd = (int32_t) ( xrs1 >> (-imm-1) );
            rd_p[i] = (int16_t) (( xrd + 1 ) >> 1);
        }else if ( imm >= 0 && imm < 16){
            rd_p[i] = (int16_t)signed_saturate(env, xrs1 << imm, 16);
        }else{
            rd_p[i] = (int16_t) 0x0000;
        }
    }

    return rd;  
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

uint32_t HELPER(sati_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{ 
    return (uint32_t)signed_saturate(env, rs1, imm + 1);
}

uint32_t HELPER(usati_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    uint32_t rd = 0;
    int32_t v1 = rs1;
    int32_t max = (1 << imm) - 1;

    if (v1 > max) {
        v1 = max;
        env->vxsat = 0x1;
    } else if (v1 < 0) {
        v1 = 0;
        env->vxsat = 0x1;
    }
    rd = v1;
    return rd;     
}

uint32_t HELPER(sslai)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    uint32_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int64_t v1 = rs1_p[0];

    rd = signed_saturate(env, v1 << imm, 32);

    return rd;     
}

uint32_t HELPER(srari_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    uint32_t rd = 0;
    int32_t v1 = rs1;

    if (imm == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (imm - 1)) + 1) >> 1;
    }

    return rd;     
}

uint32_t HELPER(ssha)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
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
uint32_t HELPER(sshar)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
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

    return (uint32_t)rd;
}

uint32_t HELPER(mseq)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    return rs1 == rs2 ? 0xFFFFFFFF : 0x00000000;
}

uint32_t HELPER(mslt)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    return (int32_t)rs1 < (int32_t)rs2 ? 0xFFFFFFFF : 0x00000000;
}

uint32_t HELPER(msltu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    return (uint32_t)rs1 < (uint32_t)rs2 ? 0xFFFFFFFF : 0x00000000;
}

uint64_t HELPER(psext_w_b)(CPURISCVState *env, uint64_t rs1)
{
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    t1 = sextract64(rs1, 32, 8);
    t2 = sextract64(rs1, 0, 8);

    return (uint64_t)((t1 << 32) | (t2 & 0xFFFFFFFF));
}

uint64_t HELPER(psext_w_h)(CPURISCVState *env, uint64_t rs1)
{
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    t1 = sextract64(rs1, 32, 16);
    t2 = sextract64(rs1, 0, 16);

    return (uint64_t)((t1 << 32) | (t2 & 0xFFFFFFFF));
}

uint64_t HELPER(psati_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;
}

uint64_t HELPER(pusati_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        int64_t max = (1 << imm) - 1;
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

uint64_t HELPER(pslli_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
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

uint64_t HELPER(psll_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
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

uint64_t HELPER(psrli_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
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

uint64_t HELPER(psrl_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
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

uint64_t HELPER(psrai_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
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

uint64_t HELPER(psra_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
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

uint64_t HELPER(psslai_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 << imm, 32);
    }

    return rd;     
}

uint64_t HELPER(psrari_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    if (imm == 0) {
        rd = rs1;
    } else {
        for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(((v1 >> (imm - 1)) + 1) >> 1);
        }
    }

    return rd;     
}

uint64_t HELPER(pssha_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
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

uint64_t HELPER(psshar_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
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

uint64_t HELPER(pmin_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pminu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pmax_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pmaxu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pmseq_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pmslt_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(pmsltu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

uint64_t HELPER(sati_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;
}

uint64_t HELPER(usati_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        int64_t max = (1 << imm) - 1;
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

uint64_t HELPER(srari_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int64_t v1 = rs1;

    if (imm == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (imm - 1)) + 1) >> 1;
    }

    return rd;  
}

uint64_t HELPER(sha)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
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

uint64_t HELPER(shar)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
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

    return (uint64_t)rd;    
}

target_ulong HELPER(ppack_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 0, 8);
        t2 = extract16(rs2_p[i], 0, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(ppackbt_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 0, 8);
        t2 = extract16(rs2_p[i], 8, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(ppacktb_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 8, 8);
        t2 = extract16(rs2_p[i], 0, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(ppackt_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 8, 8);
        t2 = extract16(rs2_p[i], 8, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;  
}

uint32_t HELPER(packbt_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t t1 = extract32(rs1, 0, 16);
    uint32_t t2 = extract32(rs2, 16, 16);
    rd = (t2 << 16) | (t1 & 0xFFFF);
    
    return rd;
}

uint32_t HELPER(packtb_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t t1 = extract32(rs1, 16, 16);
    uint32_t t2 = extract32(rs2, 0, 16);
    rd = (t2 << 16) | (t1 & 0xFFFF);
    
    return rd;
}

uint32_t HELPER(packt_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t t1 = extract32(rs1, 16, 16);
    uint32_t t2 = extract32(rs2, 16, 16);
    rd = (t2 << 16) | (t1 & 0xFFFF);
    
    return rd;
}

uint64_t HELPER(ppack_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16(rs1_p[i], 0, 16);
        t2 = extract16(rs2_p[i], 0, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(ppackbt_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16(rs1_p[i], 0, 16);
        t2 = extract16(rs2_p[2], 16, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(ppacktb_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16(rs1_p[i], 16, 16);
        t2 = extract16(rs2_p[2], 0, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(ppackt_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16(rs1_p[i], 16, 16);
        t2 = extract16(rs2_p[2], 16, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(packbt_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = extract64(rs1, 0, 32);
    uint64_t t2 = extract64(rs2, 32, 32);
    rd = (t2 << 32) | (t1 & 0xFFFFFFFF);
    
    return rd;
}

uint64_t HELPER(packtb_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = extract64(rs1, 32, 32);
    uint64_t t2 = extract64(rs2, 0, 32);
    rd = (t2 << 32) | (t1 & 0xFFFFFFFF);
    
    return rd;
}

uint64_t HELPER(packt_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = extract64(rs1, 32, 32);
    uint64_t t2 = extract64(rs2, 32, 32);
    rd = (t2 << 32) | (t1 & 0xFFFFFFFF);
    
    return rd;
}

uint64_t HELPER(zip8p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 8*(3-i), 8);
        t2 = extract64(rs2, 8*(3-i), 8);
        rd = (rd <<16) | (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(zip8hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 8*(7-i), 8);
        t2 = extract64(rs2, 8*(7-i), 8);
        rd = (rd <<16) | (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(unzip8p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 16*(3-i), 8);
        t2 = extract64(rs2, 16*(3-i), 8);
        rd = (rd <<8) | (t2 << 32) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(unzip8hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 16*(3-i)+8, 8);
        t2 = extract64(rs2, 16*(3-i)+8, 8);
        rd = (rd <<8) | (t2 << 32) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(zip16p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 16*(1-i), 16);
        t2 = extract64(rs2, 16*(1-i), 16);
        rd = (rd <<32) | (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;   
}

uint64_t HELPER(zip16hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 16*(3-i), 16);
        t2 = extract64(rs2, 16*(3-i), 16);
        rd = (rd <<32) | (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;   
}

uint64_t HELPER(unzip16p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 32*(1-i), 16);
        t2 = extract64(rs2, 32*(1-i), 16);
        rd = (rd <<16) | (t2 << 32) | (t1 & 0xFFFF);
    }

    return rd;   
}

uint64_t HELPER(unzip16hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 32*(1-i)+16, 16);
        t2 = extract64(rs2, 32*(1-i)+16, 16);
        rd = (rd << 16) | (t2 << 32) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(abs)(CPURISCVState *env, target_ulong rs1)
{
    target_long signed_rs1 = (target_long)rs1;
    return (signed_rs1 < 0) ? (0 - signed_rs1) : signed_rs1;
}

target_ulong HELPER(cls)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    target_long v = (target_long)rs1;
    #if defined(TARGET_RISCV64)
    int64_t lo_bound =  0xC000000000000000;
    int64_t hi_bound =  0x3FFFFFFFFFFFFFFF;
    #elif defined(TARGET_RISCV32)
    int32_t lo_bound =  0xC0000000;
    int32_t hi_bound =  0x3FFFFFFF;
    #endif

    while ( rd < TARGET_LONG_BITS && v >= lo_bound && v <= hi_bound ) {
        rd = rd + 1;
        v = v << 1;
    }

    return rd;
}

target_ulong HELPER(rev)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    for (int i = 0; i < TARGET_LONG_BITS; i++){
        rd = (rd << 1) | (rs1 & 1);
        rs1 >>= 1;
    }

    return rd;
}

uint64_t HELPER(rev16)(CPURISCVState *env, uint64_t rs1)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;

    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 16*i, 16);
        rd = (rd << 16) | (t1 & 0xFFFF);        
    }

    return rd;
}

target_ulong HELPER(slx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    int shamt = (TARGET_LONG_BITS == 32) ? (rs2 & 0x1F) : (rs2 & 0x3F);
    target_ulong xrs1 = 0; 
    target_ulong xrd = 0; 
    if(shamt <= TARGET_LONG_BITS){
        xrs1 = rs1 >> (TARGET_LONG_BITS - shamt);
        xrd = (rd << shamt) + xrs1;
    }else{
        xrd = rs1 << (shamt - TARGET_LONG_BITS);
    }

    return xrd;
}

target_ulong HELPER(srx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    int shamt = (TARGET_LONG_BITS == 32) ? (rs2 & 0x1F) : (rs2 & 0x3F);
    target_ulong xrs1 = 0; 
    target_ulong xrd = 0; 
    if(shamt <= TARGET_LONG_BITS){
        xrs1 = rs1 << (TARGET_LONG_BITS - shamt);
        xrd = (rd >> shamt) + xrs1;
    }else{
        xrd = rs1 >> (shamt - TARGET_LONG_BITS);
    }

    return xrd;
}

target_ulong HELPER(mvm)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    return (~rs2 & rd) | (rs2 & rs1);
}

target_ulong HELPER(mvmn)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    return (~rs2 & rs1) | (rs2 & rd);
}

target_ulong HELPER(merge)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    return (~rd & rs1) | (rd & rs2);
}

uint64_t HELPER(absw)(CPURISCVState *env, uint64_t rs1)
{
    int32_t rs1_w = rs1 & 0xFFFFFFFF;
    return (rs1_w < 0) ? (0 - rs1_w) : rs1_w;
}

uint64_t HELPER(clsw)(CPURISCVState *env, uint64_t rs1)
{
    int32_t rs1_w = rs1 & 0xFFFFFFFF;
    int c = 0;
    while( c < 32 && rs1_w >= 0xC0000000 && rs1_w <= 0x3FFFFFFF){
        c = c + 1;
        rs1_w = rs1_w << 1;
    }

    return (uint64_t) c;
}

uint64_t HELPER(pwadd_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwadda_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwaddu_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwaddau_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwsub_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsuba_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubu_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubau_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwslli_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsll_bs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint32_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 4; i++) {
        v1 = (uint32_t) rs1_p[i];
        rd_p[i] = (uint16_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(pwslai_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    int16_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsla_bs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    int32_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 4; i++) {
        v1 = (int32_t) rs1_p[i];
        rd_p[i] = (int16_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(pwadd_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwadda_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwaddu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwaddau_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwsub_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsuba_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubau_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwslli_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsll_hs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 2; i++) {
        v1 = (uint64_t) rs1_p[i];
        rd_p[i] = (uint32_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(pwslai_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int32_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsla_hs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 2; i++) {
        v1 = (int64_t) rs1_p[i];
        rd_p[i] = (int32_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(wadd)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)((int64_t)rs1 + (int64_t)rs2);
}

uint64_t HELPER(wadda)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return (uint64_t)((int64_t)rd+(int64_t)rs1 + (int64_t)rs2);
}

uint64_t HELPER(waddu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)rs1 + (uint64_t)rs2;
}

uint64_t HELPER(waddau)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return rd + rs1 + rs2;
}

uint64_t HELPER(wsub)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)((int64_t)rs1 - (int64_t)rs2);
}

uint64_t HELPER(wsuba)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return (uint64_t)( (int64_t)rd + (int64_t)rs1 - (int64_t)rs2 );
}

uint64_t HELPER(wsubu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)rs1 - (uint64_t)rs2;
}

uint64_t HELPER(wsubau)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return rd + rs1 - rs2;
}

uint64_t HELPER(wslli)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    return rs1 << imm;
}

uint64_t HELPER(wsll)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint8_t imm = rs2 & 0x3F; //extract rs2[5..0]
    return rs1 << imm;
}

uint64_t HELPER(wslai)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    return (int64_t)rs1 << imm;
}

uint64_t HELPER(wsla)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint8_t imm = rs2 & 0x3F; //extract rs2[5..0]
    return (int64_t)rs1 << imm;
}

uint64_t HELPER(wzip8p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 8*(3-i), 8);
        t2 = extract64(rs2, 8*(3-i), 8);
        rd = (rd <<16) | (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(wzip16p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 16*(1-i), 16);
        t2 = extract64(rs2, 16*(1-i), 16);
        rd = (rd <<32) | (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint32_t HELPER(predsum_dbs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    int32_t rd = rs2;
    int8_t *rs1_p = (int8_t *)&rs1_l;
    int32_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (int8_t *)&rs1_h;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (uint32_t)rd;
}

uint32_t HELPER(predsumu_dbs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    uint32_t rd = rs2;
    uint8_t *rs1_p = (uint8_t *)&rs1_l;
    uint32_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (uint8_t *)&rs1_h;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

uint32_t HELPER(predsum_dhs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    int32_t rd = rs2;
    int16_t *rs1_p = (int16_t *)&rs1_l;
    int16_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (int16_t *)&rs1_h;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (uint32_t)rd;
}

uint32_t HELPER(predsumu_dhs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    uint32_t rd = rs2;
    uint16_t *rs1_p = (uint16_t *)&rs1_l;
    uint32_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (uint16_t *)&rs1_h;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

uint32_t HELPER(pnsrli_b)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        uint8_t result = (s1_h >> (shamt & 0xF)) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrai_b)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int32_t s1_h_s32 = (int16_t)s1_h;
        int32_t s1_h_s24 = (s1_h_s32 << 8) >> 8;
        uint8_t result = s1_h_s24 >> (shamt & 0xF) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrari_b)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
     uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int32_t s1_h_s32 = (int16_t)s1_h;
        int32_t s1_h_s24 = (s1_h_s32 << 8) >> 8;
        uint32_t shx_25bit = ((uint32_t)s1_h_s24 << 1);
        uint32_t shx = (shx_25bit >> (shamt & 0x1F)) & 0x1FF;
        uint8_t result = ((shx + 1) >> 1) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrl_bs)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        uint32_t s1_h_z32 = (uint32_t)s1_h;
        uint8_t result = (s1_h_z32 >> (shamt & 0x1F)) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsra_bs)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int64_t s1_h_s64 = (int16_t)s1_h;
        s1_h_s64 = (s1_h_s64 << 24) >> 24;
        uint8_t result = s1_h_s64 >> (shamt & 0x1F) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrar_bs)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int64_t s1_h_s64 = (int16_t)s1_h;
        int64_t s1_h_s40 = (s1_h_s64 << 24) >> 24;
        uint64_t shx_41bit = ((uint64_t)s1_h_s40 << 1);
        uint64_t shx = (shx_41bit >> (shamt & 0x1F)) & 0x1FF;
        uint8_t result = ((shx + 1) >> 1) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}