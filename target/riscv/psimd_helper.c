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
