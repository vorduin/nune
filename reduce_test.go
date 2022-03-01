// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"

	"github.com/vorduin/nune"
)

func BenchmarkMin(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Min()
	})
}

func BenchmarkMax(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Max()
	})
}

func BenchmarkMean(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Mean()
	})
}

func BenchmarkSum(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Sum()
	})
}

func BenchmarkProd(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Prod()
	})
}
