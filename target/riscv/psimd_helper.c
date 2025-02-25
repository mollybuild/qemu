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

target_ulong helper_padd_h(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        rd_p[i] = rs1_p[i] + rs2_p[i];
    }

    return rd;
}
