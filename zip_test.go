// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"

	"github.com/vorduin/nune"
)

func BenchmarkAdd(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Add(tensor)
	})
}
func BenchmarkSub(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Sub(tensor)
	})
}

func BenchmarkMul(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Mul(tensor)
	})
}

func BenchmarkDiv(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Div(tensor)
	})
}
