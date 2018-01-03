__kernel void serial(  __global float *a,
                       const unsigned int n)
{
    uint glob_id = get_global_id(0);
	uint glob_size = get_global_size(0);
	uint index = glob_id;

	float acc = 0;
  // Loop sequentially over chunks of input vector
  while (index < n) {
    acc += a[index];
    index += glob_size;
  }
  
	a[glob_id] = acc;
}