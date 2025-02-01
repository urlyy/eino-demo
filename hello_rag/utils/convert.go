package utils

func SliceFloat64To32(f64Slice []float64) []float32 {
	// 创建一个[]float32类型的切片，长度与f64Slice相同
	f32Slice := make([]float32, len(f64Slice))

	// 遍历f64Slice，将每个元素转换为float32并存储到f32Slice中
	for i, v := range f64Slice {
		f32Slice[i] = float32(v)
	}
	return f32Slice
}
