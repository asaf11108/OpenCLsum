__kernel void sum(  __global float *a,
                       __global float *sum,
                       const unsigned int n)
{
    int glob_id = get_global_id(0);
	int loc_id = get_local_id(0);
	int loc_size = get_local_size(0);
    int group_id = get_group_id(0);

	for(unsigned int s = loc_size / 2 ; s>0 ; s >>= 1 ){
		if(loc_id < s){
			a[glob_id] += a[glob_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(loc_id == 0){
		//printf("%d: %f\n", group_id, a[glob_id]);
		sum[group_id] = a[glob_id];
	}
}