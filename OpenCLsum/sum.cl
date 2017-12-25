__kernel void sum(  __global float *a,
                       __global float *sum,
					   __local float *mem,
                       const unsigned int n)
{
    uint glob_id = get_global_id(0);
	uint loc_id = get_local_id(0);
	uint loc_size = get_local_size(0);
    uint group_id = get_group_id(0);
	
	mem[loc_id] = a[glob_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = loc_size / 2 ; s>0 ; s >>= 1 ){
		if(loc_id < s){
			//a[glob_id] += a[glob_id + s];
			mem[loc_id] += mem[loc_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(loc_id == 0){
		//printf("%d: %f\n", group_id, a[glob_id]);
		sum[group_id] = mem[0];
	}
}