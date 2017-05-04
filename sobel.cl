string source = "__kernel void sobel(__global const float *image,\n"
    "__global float *result)\n"
    "{\n"
    "   int width = 10;\n"
    "   int lwidth = 6;\n"
    "\n"
    "   int gidx = get_global_id(0);\n"
    "   int gidy = get_global_id(1);\n"
    "\n"
    "   int lidx = get_local_id(0);\n"
    "   int lidy = get_local_id(1);\n"
    "\n"
    "   __local float buf[36];\n"
    "   __local float tmp[36];\n"
    "   __local float Ix[36];\n"
    "   __local float Iy[36];\n"
    "\n"
    "   buf[(lidy + 1) * lwidth + (lidx + 1)] = image[(gidy + 1) * width + (gidx + 1)];\n"
    "   if (lidy == 0) {\n"
    "       buf[lidy + (lidx + 1)]                    = image[gidy * width + (gidx + 1)];\n"
    "       buf[(lidy + 4 + 1) * lwidth + (lidx + 1)] = image[(gidy + 4 + 1) * width + (gidx + 1)];\n"
    "   }\n"
    "   if (lidx == 0) {\n"
    "       buf[(lidy + 1) * lwidth + lidx]           = image[(gidy + 1) * width + gidx];\n"
    "       buf[(lidy + 1) * lwidth + (lidx + 4 + 1)] = image[(gidy + 1) * width + (gidx + 4 + 1)];\n"
    "   }\n"
    "   barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "   tmp[(lidy + 1) * lwidth + (lidx + 1)] = buf[(lidy + 1) * lwidth + lidx] * -1.0 + buf[(lidy + 1) * lwidth + (lidx + 1)] * 0.0 + buf[(lidy + 1) * lwidth + (lidx + 2)] * 1.0;\n"
    "   barrier(CLK_LOCAL_MEM_FENCE);\n"
    "   Ix[(lidy + 1) * lwidth + (lidx + 1)] = tmp[lidy * lwidth + (lidx + 1)] * 1.0 + tmp[(lidy + 1) * lwidth + (lidx + 1)] * 2.0 + tmp[(lidy + 2) * lwidth + (lidx + 1)] * 1.0;\n"
    "   \n"
    "   tmp[(lidy + 1) * lwidth + (lidx + 1)] = buf[(lidy + 1) * lwidth + lidx] * 1.0 + buf[(lidy + 1) * lwidth + (lidx + 1)] * 2.0 + buf[(lidy + 1) * lwidth + (lidx + 2)] * 1.0;\n"
    "   barrier(CLK_LOCAL_MEM_FENCE);\n"
    "   Iy[(lidy + 1) * lwidth + (lidx + 1)] = tmp[lidy * lwidth + (lidx + 1)] * -1.0 + tmp[(lidy + 1) * lwidth + (lidx + 1)] * 0.0 + tmp[(lidy + 2) * lwidth + (lidx + 1)] * 1.0;\n"
    "   barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "   float ix = Ix[(lidy + 1) * lwidth + (lidx + 1)];\n"
    "   float iy = Iy[(lidy + 1) * lwidth + (lidx + 1)];\n"
    "   //result[(gidy + 1) * width + (gidx + 1)] = buf[(lidy + 1) * lwidth + (lidx + 1)];\n"
    "   result[(gidy + 1) * width + (gidx + 1)] = sqrt((ix * ix) + (iy * iy));\n"
    "   \n"
    "}\n";
