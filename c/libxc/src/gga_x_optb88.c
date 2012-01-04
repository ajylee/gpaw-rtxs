/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
  
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
  
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "util.h"

#define XC_GGA_X_OPTB88       125 /* optB88 for optB88-vdW */

typedef struct{
  FLOAT beta, gamma;
} gga_x_optb88_params;

static void 
gga_x_optb88_init(void *p_)
{
  XC(gga_type) *p = (XC(gga_type) *)p_;

  assert(p->params == NULL);
  p->params = malloc(sizeof(gga_x_optb88_params));

  /* value of beta and gamma in reparametrized Becke 88 functional */
  XC(gga_x_optb88_set_params)(p, 0.00336865923905927, 6.98131700797731);
}

static void gga_x_optb88_end(void *p_)
{
  XC(gga_type) *p = (XC(gga_type) *)p_;

  assert(p->params != NULL);
  free(p->params);
  p->params = NULL;
}

void XC(gga_x_optb88_set_params)(XC(gga_type) *p, FLOAT beta, FLOAT gamma)
{
  gga_x_optb88_params *params;

  assert(p->params != NULL);
  params = (gga_x_optb88_params *) (p->params);

  params->beta = beta;
  params->gamma = gamma;
}

static inline void 
func(const XC(gga_type) *p, FLOAT x, FLOAT *f, FLOAT *dfdx, FLOAT *ldfdx, FLOAT *d2fdx2)
{
  FLOAT f1, f2, df1, df2, d2f1, d2f2;
  FLOAT beta, gamma;

  assert(p->params != NULL);
  beta = ((gga_x_optb88_params *) (p->params))->beta;
  gamma = ((gga_x_optb88_params *) (p->params))->gamma;

  f1 = beta/X_FACTOR_C*x*x;
  f2 = 1.0 + gamma*beta*x*asinh(x);
  *f = 1.0 + f1/f2;
 
  if(dfdx==NULL && d2fdx2==NULL) return; /* nothing else to do */

  df1 = 2.0*beta/X_FACTOR_C*x;
  df2 = gamma*beta*(asinh(x) + x/sqrt(1.0 + x*x));
  if(dfdx!=NULL){
    *dfdx = (df1*f2 - f1*df2)/(f2*f2);
    *ldfdx= beta/X_FACTOR_C;
  }

  if(d2fdx2==NULL) return; /* nothing else to do */

  d2f1 = 2.0*beta/X_FACTOR_C;
  d2f2 = gamma*beta*(2.0 + x*x)/pow(1.0 + x*x, 3.0/2.0);

  *d2fdx2 = (2.0*f1*df2*df2 + d2f1*f2*f2 - f2*(2.0*df1*df2 + f1*d2f2))/(f2*f2*f2);
}

#include "work_gga_x.c"

const XC(func_info_type) XC(func_info_gga_x_optb88) = {
  XC_GGA_X_OPTB88,
  XC_EXCHANGE,
  "optB88 exchange for optB88-vdW",
  XC_FAMILY_GGA,
  "J Klimes, DR Bowler, and A Michaelides, J. Phys.: Condens. Matter 22, 022201 (2010)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC | XC_PROVIDES_FXC,
  gga_x_optb88_init, 
  gga_x_optb88_end, 
  NULL,
  work_gga_x
};
