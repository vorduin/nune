// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"

	"github.com/vorduin/nune"
)

func BenchmarkAbs(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Abs()
	})
}

func BenchmarkAcos(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Acos()
	})
}

func BenchmarkAcosh(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Acosh()
	})
}

func BenchmarkAsin(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Asin()
	})
}

func BenchmarkAsinh(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Asinh()
	})
}

func BenchmarkAtan(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Atan()
	})
}

func BenchmarkAtan2(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Atan2(0)
	})
}

func BenchmarkAtanh(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Atanh()
	})
}

func BenchmarkCbrt(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Cbrt()
	})
}

func BenchmarkCeil(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Ceil()
	})
}

func BenchmarkCopysign(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Copysign(-1)
	})
}

func BenchmarkCos(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Cos()
	})
}

func BenchmarkCosh(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Cosh()
	})
}

func BenchmarkDim(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Dim(0)
	})
}

func BenchmarkErf(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Erf()
	})
}

func BenchmarkErfc(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Erfc()
	})
}

func BenchmarkErfcinv(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Erfcinv()
	})
}

func BenchmarkErfinv(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Erfinv()
	})
}

func BenchmarkExp(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Exp()
	})
}

func BenchmarkExp2(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Exp2()
	})
}

func BenchmarkExpm1(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Expm1()
	})
}

func BenchmarkFMA(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.FMA(0, 1)
	})
}

func BenchmarkFloor(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Floor()
	})
}

func BenchmarkGamma(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Gamma()
	})
}

func BenchmarkIlogb(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Ilogb()
	})
}

func BenchmarkInv(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Inf()
	})
}

func BenchmarkJ0(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.J0()
	})
}

func BenchmarkJ1(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.J1()
	})
}

func BenchmarkJn(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Jn(2)
	})
}

func BenchmarkLog(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Log()
	})
}

func BenchmarkLog10(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Log10()
	})
}

func BenchmarkLog1p(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Log1p()
	})
}

func BenchmarkLog2(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Log2()
	})
}

func BenchmarkLogb(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Logb()
	})
}

func BenchmarkMod(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Mod(2)
	})
}

func BenchmarkNaN(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.NaN()
	})
}

func BenchmarkNextafter(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Nextafter(0)
	})
}

func BenchmarkNextafter32(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Nextafter32(0)
	})
}

func BenchmarkPow(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Pow(2)
	})
}

func BenchmarkPow10(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Pow10(2)
	})
}

func BenchmarkRemainder(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Remainder(2)
	})
}

func BenchmarkRound(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Round()
	})
}

func BenchmarkRoundToEven(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.RoundToEven()
	})
}

func BenchmarkSin(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Sin()
	})
}

func BenchmarkSinh(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Sinh()
	})
}

func BenchmarkSqrt(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Sqrt()
	})
}

func BenchmarkTan(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Tan()
	})
}

func BenchmarkTanh(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Tanh()
	})
}

func BenchmarkTrunc(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Trunc()
	})
}

func BenchmarkY0(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Y0()
	})
}

func BenchmarkY1(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Y1()
	})
}

func BenchmarkYn(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		tensor.Yn(2)
	})
}
