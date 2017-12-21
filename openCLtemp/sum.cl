__kernel void sum(  __global float *a,
                       __global float *sum,
                       const unsigned int n)
{
    int glob_id = get_global_id(0);
	int loc_id = get_local_id(0);
	int loc_size = get_local_size(0);
    int group_id = get_group_id(0);

if(glob_id == 0){
	sum[0] = sum[0] + a[0];
	printf("sum: %f\n", a[0]);
}
}