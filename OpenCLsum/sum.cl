#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void sum(  __global float *a,
                       __global double *sum,
                       const unsigned int n)
{
    //Get our global thread ID
    int id = get_global_id(0);
	
    //Make sure we do not go out of bounds
    if (id == 0){ 
		sum[0]=0;
		for (int i=0;i<n;i++){
			sum[0]+=a[i];
		 }
	}
}