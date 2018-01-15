
__kernel void preparation(global int *d_labels, global int *d_pixels, int bgc, int width, int height) {
	  int x = get_global_id(0);
	  int y = get_global_id(1);
	  int cur_pixel = y * width + x;

	  if (x >= width || y >= height) return;

	  if (d_pixels[cur_pixel] == bgc) 
		d_labels[cur_pixel] = -1;
	  else
		d_labels[cur_pixel] = cur_pixel;
}

__kernel void propagation(global int *d_labels, global int *d_pixels, global int *d_passes, int cur_pass, int width, int height) {
	  int x = get_global_id(0);
	  int y = get_global_id(1);

	  if (x >= width || y >= height)
		return;

	  if (d_passes[cur_pass-1] == 0)
		return;

	  int cur_pixel = y * width + x;
	  int g = d_labels[cur_pixel];
	  int og = g;

	  if (g == -1)
		return;

	  for(int yy=-1;yy<=1;yy++) {
		for(int xx=-1;xx<=1;xx++) {
		  if (0 <=  x + xx &&  x + xx < width && 0 <=  y + yy &&  y + yy < height) {
		const int p1 = (y + yy) * width + x + xx, s = d_labels[p1];
		if (s != -1 && s < g) g = s;
		  }
		}
	  }

	  g = d_labels[d_labels[d_labels[g]]];

	  if (g != og) {
		atomic_min(&d_labels[og], g);
		atomic_min(&d_labels[cur_pixel], g);
		d_passes[cur_pass] = 1;
	  }
}