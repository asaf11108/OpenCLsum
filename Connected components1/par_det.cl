
__kernel void preparation(global int *d_pixels, global int *d_labels, int width, int height) {
	  int x = get_global_id(0);
	  int y = get_global_id(1);
	  int cur_pixel = y * width + x;

	  if (x >= width || y >= height)
		return;
		
		//background color is black
	  if (d_pixels[cur_pixel] == 0) 
		d_labels[cur_pixel] = -1;
	  else
		d_labels[cur_pixel] = cur_pixel;
}

__kernel void propagation(global int *d_pixels, global int *d_labels, global bool *d_passes, int width, int height, int cur_pass) {
	  int x = get_global_id(0);
	  int y = get_global_id(1);

	  if (x >= width || y >= height)
		return;

	  if (d_passes[cur_pass] == false)
		return;

	  int cur_pixel = y * width + x;
	  int min_cur_pixel = d_labels[cur_pixel];
	  if (min_cur_pixel == -1)
		return;

	  int old_min_cur_pixel = min_cur_pixel;

	  for(int sub_y = -1; sub_y <= 1; sub_y++) {
		for(int sub_x = -1; sub_x <= 1; sub_x++) {
		  if (0 <=  x + sub_x &&  x + sub_x < width && 0 <=  y + sub_y &&  y + sub_y < height) {
			int p = (y + sub_y) * width + x + sub_x;
			int min_sub = d_labels[p];
			if (min_sub != -1 && min_sub < min_cur_pixel)
				min_cur_pixel = min_sub;
		  }
		}
	  }
	  //printf("cur_pass: %d", cur_pass);
	  min_cur_pixel = d_labels[d_labels[d_labels[min_cur_pixel]]];

	  if (min_cur_pixel != old_min_cur_pixel) {
		atomic_min(&d_labels[old_min_cur_pixel], min_cur_pixel);
		atomic_min(&d_labels[cur_pixel], min_cur_pixel);
		d_passes[cur_pass+1] = true;
	  }
}