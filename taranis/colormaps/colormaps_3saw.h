/*
 * Ncview by David W. Pierce.  A visual netCDF file viewer.
 * Copyright (C) 1993 through 2010 David W. Pierce
 *
 * This program  is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as 
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License, version 3, for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * David W. Pierce
 * 6259 Caminito Carrean
 * San Diego, CA   92122
 * pierce@cirrus.ucsd.edu
 */

static int cmap_3saw[] = {
	0,255,255, 1,251,255, 2,246,254, 3,241,253, 4,236,252, 5,231,251, 6,226,250, 7,221,249, 
	8,216,248, 9,211,247, 10,206,246, 11,201,245, 12,196,244, 13,191,243, 14,186,242, 15,181,241, 
	16,176,240, 17,171,239, 18,166,238, 19,161,237, 20,156,236, 21,151,235, 22,146,234, 23,141,233, 
	24,136,232, 25,131,231, 26,126,230, 27,121,229, 28,116,228, 29,111,227, 30,106,226, 31,101,225, 
	32,96,224, 33,91,223, 34,86,222, 35,81,221, 36,76,220, 37,71,219, 38,66,218, 39,61,217, 
	40,56,216, 41,51,215, 42,46,214, 43,41,213, 44,36,212, 45,31,211, 46,26,210, 47,21,209, 
	48,16,208, 49,11,207, 50,6,206, 51,1,205, 52,252,204, 53,247,203, 54,242,202, 55,237,201, 
	56,232,200, 57,227,199, 58,222,198, 59,217,197, 60,212,196, 61,207,195, 62,202,194, 63,197,193, 
	64,192,192, 65,187,191, 66,182,190, 67,177,189, 68,172,188, 69,167,187, 70,162,186, 71,157,185, 
	72,152,184, 73,147,183, 74,142,182, 75,137,181, 76,132,180, 77,127,179, 78,122,178, 79,117,177, 
	80,112,176, 81,107,175, 82,102,174, 83,97,173, 84,92,172, 85,87,171, 86,82,170, 87,77,169, 
	88,72,168, 89,67,167, 90,62,166, 91,57,165, 92,52,164, 93,47,163, 94,42,162, 95,37,161, 
	96,32,160, 97,27,159, 98,22,158, 99,17,157, 100,12,156, 101,7,155, 102,2,154, 103,253,153, 
	104,248,152, 105,243,151, 106,238,150, 107,233,149, 108,228,148, 109,223,147, 110,218,146, 111,213,145, 
	112,208,144, 113,203,143, 114,198,142, 115,193,141, 116,188,140, 117,183,139, 118,178,138, 119,173,137, 
	120,168,136, 121,163,135, 122,158,134, 123,153,133, 124,148,132, 125,143,131, 126,138,130, 127,133,129, 
	128,128,128, 129,123,127, 130,118,126, 131,113,125, 132,108,124, 133,103,123, 134,98,122, 135,93,121, 
	136,88,120, 137,83,119, 138,78,118, 139,73,117, 140,68,116, 141,63,115, 142,58,114, 143,53,113, 
	144,48,112, 145,43,111, 146,38,110, 147,33,109, 148,28,108, 149,23,107, 150,18,106, 151,13,105, 
	152,8,104, 153,3,103, 154,254,102, 155,249,101, 156,244,100, 157,239,99, 158,234,98, 159,229,97, 
	160,224,96, 161,219,95, 162,214,94, 163,209,93, 164,204,92, 165,199,91, 166,194,90, 167,189,89, 
	168,184,88, 169,179,87, 170,174,86, 171,169,85, 172,164,84, 173,159,83, 174,154,82, 175,149,81, 
	176,144,80, 177,139,79, 178,134,78, 179,129,77, 180,124,76, 181,119,75, 182,114,74, 183,109,73, 
	184,104,72, 185,99,71, 186,94,70, 187,89,69, 188,84,68, 189,79,67, 190,74,66, 191,69,65, 
	192,64,64, 193,59,63, 194,54,62, 195,49,61, 196,44,60, 197,39,59, 198,34,58, 199,29,57, 
	200,24,56, 201,19,55, 202,14,54, 203,9,53, 204,4,52, 205,255,51, 206,250,50, 207,245,49, 
	208,240,48, 209,235,47, 210,230,46, 211,225,45, 212,220,44, 213,215,43, 214,210,42, 215,205,41, 
	216,200,40, 217,195,39, 218,190,38, 219,185,37, 220,180,36, 221,175,35, 222,170,34, 223,165,33, 
	224,160,32, 225,155,31, 226,150,30, 227,145,29, 228,140,28, 229,135,27, 230,130,26, 231,125,25, 
	232,120,24, 233,115,23, 234,110,22, 235,105,21, 236,100,20, 237,95,19, 238,90,18, 239,85,17, 
	240,80,16, 241,75,15, 242,70,14, 243,65,13, 244,60,12, 245,55,11, 246,50,10, 247,45,9, 
	248,40,8, 249,35,7, 250,30,6, 251,25,5, 252,20,4, 253,15,3, 254,10,2, 255,5,1};
