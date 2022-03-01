// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

// import (
// 	"testing"

// 	"github.com/vorduin/nune"
// )

// func BenchmarkCast1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1)

// 	benchmarkMicro(b, func() {
// 		nune.Cast[float32](tensor)
// 	})
// }

// func BenchmarkClone1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1)

// 	benchmarkMicro(b, func() {
// 		tensor.Clone()
// 	})
// }

// func BenchmarkReshape1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1)

// 	benchmarkMicro(b, func() {
// 		tensor.Reshape(100, 200, 500)
// 	})
// }

// func BenchmarkIndex1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1).Reshape(10, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Index(50)
// 	})
// }

// func BenchmarkSlice1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1).Reshape(10, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Slice(25, 75)
// 	})
// }

// func BenchmarkBroadcast1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 200, 1).Reshape(200, 1)

// 	benchmarkMicro(b, func() {
// 		tensor.Broadcast(10, 200, 500)
// 	})
// }

// func BenchmarkReverse1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1)

// 	benchmarkMicro(b, func() {
// 		tensor.Reverse()
// 	})
// }

// func BenchmarkFlip1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1).Reshape(10, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Flip(1)
// 	})
// }

// func BenchmarkRepeat1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e5, 1).Reshape(200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Repeat(10)
// 	})
// }

// func BenchmarkPermute1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1).Reshape(10, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Permute(2, 1, 0)
// 	})
// }

// func BenchmarkCat1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 5e5, 1).Reshape(5, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Cat(tensor, 0)
// 	})
// }

// func BenchmarkStack1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 5e5, 1).Reshape(5, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Stack(tensor, 0)
// 	})
// }

// func BenchmarkSqueeze1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1).Reshape(10, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Squeeze(1)
// 	})
// }

// func BenchmarkUnsqueeze1e6(b *testing.B) {
// 	tensor := nune.Range[float64](0, 1e6, 1).Reshape(10, 1, 200, 500)

// 	benchmarkMicro(b, func() {
// 		tensor.Unsqueeze(1)
// 	})
// }
