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
#include <assert.h>
#include "util.h"

#define XC_GGA_X_C09          126 /* Cooper 09 exchange */

static inline void 
func(const XC(gga_type) *p, FLOAT x, FLOAT *f, FLOAT *dfdx, FLOAT *ldfdx, FLOAT *d2fdx2)
{
  static const FLOAT kappa = 1.245;
  static const FLOAT mu = 0.0617;
  static const FLOAT alpha = 0.0483;

  FLOAT ss, arg, ex1, ex2;

  ss = X2S*x;

  arg = alpha*ss*ss;
  ex1 = exp(-arg);
  ex2 = exp(-arg/2.0);

  *f = 1.0 + mu*ss*ss*ex1 + kappa - kappa*ex2;

  if(dfdx==NULL && d2fdx2==NULL) return; /* nothing else to do */

  if(dfdx!=NULL){
    *dfdx  = X2S*ss*ex2*(2.0*mu*ex2 - 2.0*mu*arg*ex2 + kappa*alpha);
    *ldfdx = X2S*X2S*(mu + kappa*alpha/2.0);
  }

  if(d2fdx2==NULL) return; /* nothing else to do */
  *d2fdx2 = X2S*X2S*2.0*mu*ex1*(1.0 - 5.0*arg + 2.0*arg*arg) + kappa*alpha*ex2*(1.0 - arg);
}

#include "work_gga_x.c"

const XC(func_info_type) XC(func_info_gga_x_c09) = {
  XC_GGA_X_C09,
  XC_EXCHANGE,
  "Cooper 09",
  XC_FAMILY_GGA,
  "V. Cooper, PRB 81, 161104(R) (2010)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC | XC_PROVIDES_FXC,
  NULL, NULL, NULL,
  work_gga_x
};
