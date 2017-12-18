__kernel void sum(  __global float *a,
                       __global float *sum,
                       const unsigned int n)
{
    //myid - global thread index
	//tid - local block index

	//Get our global thread ID
    int glob_id = get_global_id(0);
	int loc_id = get_local_id(0);
    //Make sure we do not go out of bounds
    
	for(unsigned int s = 64 / 2 ; s>0 ; s >>= 1 ){
		if(loc_id < s){
			a[glob_id] += a[glob_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(loc_id == 0){
		int group_id = get_group_id(0);
		for(unsigned int s = get_num_groups(0) / 2 ; s>0 ; s >>= 1 ){
			if(group_id < s)
				a[group_id*64] += a[group_id*64+s*64];
			barrier(CLK_GLOBAL_MEM_FENCE);
		}

	}
	if(glob_id == 0)
		sum[0] = a[0];
}