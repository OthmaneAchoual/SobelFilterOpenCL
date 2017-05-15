__kernel void sobel(__global const float *image,
    __global float *result)
    {
       int width = 10;
       int lwidth = 6;
    
       int gidx = get_global_id(0);
       int gidy = get_global_id(1);
    
       int lidx = get_local_id(0);
       int lidy = get_local_id(1);
    
       __local float buf[36];
       __local float tmp[36];
       __local float Ix[36];
       __local float Iy[36];
    
       buf[(lidy + 1) * lwidth + (lidx + 1)] = image[(gidy + 1) * width + (gidx + 1)];
       if (lidy == 0) {
           buf[lidy + (lidx + 1)]                    = image[gidy * width + (gidx + 1)];
           buf[(lidy + 4 + 1) * lwidth + (lidx + 1)] = image[(gidy + 4 + 1) * width + (gidx + 1)];
       }
       if (lidx == 0) {
           buf[(lidy + 1) * lwidth + lidx]           = image[(gidy + 1) * width + gidx];
           buf[(lidy + 1) * lwidth + (lidx + 4 + 1)] = image[(gidy + 1) * width + (gidx + 4 + 1)];
       }
       barrier(CLK_LOCAL_MEM_FENCE);
    
       tmp[(lidy + 1) * lwidth + (lidx + 1)] = buf[(lidy + 1) * lwidth + lidx] * -1.0 + buf[(lidy + 1) * lwidth + (lidx + 1)] * 0.0 + buf[(lidy + 1) * lwidth + (lidx + 2)] * 1.0;
       barrier(CLK_LOCAL_MEM_FENCE);
       Ix[(lidy + 1) * lwidth + (lidx + 1)] = tmp[lidy * lwidth + (lidx + 1)] * 1.0 + tmp[(lidy + 1) * lwidth + (lidx + 1)] * 2.0 + tmp[(lidy + 2) * lwidth + (lidx + 1)] * 1.0;
    
       tmp[(lidy + 1) * lwidth + (lidx + 1)] = buf[(lidy + 1) * lwidth + lidx] * 1.0 + buf[(lidy + 1) * lwidth + (lidx + 1)] * 2.0 + buf[(lidy + 1) * lwidth + (lidx + 2)] * 1.0;
       barrier(CLK_LOCAL_MEM_FENCE);
       Iy[(lidy + 1) * lwidth + (lidx + 1)] = tmp[lidy * lwidth + (lidx + 1)] * -1.0 + tmp[(lidy + 1) * lwidth + (lidx + 1)] * 0.0 + tmp[(lidy + 2) * lwidth + (lidx + 1)] * 1.0;
       barrier(CLK_LOCAL_MEM_FENCE);
    
       float ix = Ix[(lidy + 1) * lwidth + (lidx + 1)];
       float iy = Iy[(lidy + 1) * lwidth + (lidx + 1)];
       //result[(gidy + 1) * width + (gidx + 1)] = buf[(lidy + 1) * lwidth + (lidx + 1)];
       result[(gidy + 1) * width + (gidx + 1)] = sqrt((ix * ix) + (iy * iy));
    
    }
