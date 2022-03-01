// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"sync"
	"github.com/vorduin/slices"
)

// handleZip processes an elementwise operation accordingly.
func handleZip[T Number](lhs, rhs Tensor[T], f func(T, T) T, nCPU int) {
	if lhs.Rank() == 0 {
		lhs.Ravel()[0] = f(lhs.Ravel()[0], rhs.Ravel()[0])
	}

	var wg sync.WaitGroup

	for i := 0; i < nCPU; i++ {
		min := (i * lhs.Size(0) / nCPU)
		max := ((i + 1) * lhs.Size(0)) / nCPU

		wg.Add(1)
		go func(t1, t2 Tensor[T]) {
			it1, it2 := t1.Iter(), t2.Iter()
			for j := 0; j < it1.Size(); j++ {
				n1, _ := it1.Next()
				n2, _ := it2.Next()

				n1.Ravel()[0] = f(n1.Ravel()[0], n2.Ravel()[0])
			}

			wg.Done()
		}(lhs.Slice(min, max), rhs.Slice(min, max))
	}

	wg.Wait()
}

// midwayBroadcast adjusts the first shape so that the second shape might
// be broadcastable to it. For example, for the shape [4, 1] to be
// broadcastable with the shape [3], this function returns that
// the broadcasting shape should be [4, 3].
func midwayBroadcast(s1, s2 []int) []int {
	var s []int

	if len(s1) < len(s2) {
		s = slices.WithLen[int](len(s2))
		for i := 0; i < len(s2)-len(s1); i++ {
			s[i] = 1
		}
		copy(s[len(s2)-len(s1):], s1)
	} else {
		s = s1
	}

	for i := 0; i < len(s2); i++ {
		if s[i] != s2[i] && s[i] == 1 {
			s[i] = s2[i]
		}
	}

	return s
}

// Zip performs an elementwise operation
// between other and this Tensor.
func (t Tensor[T]) Zip(other any, f func(T, T) T) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	o := From[T](other)
	if o.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			t.Err = o.Err
			return t
		}
	}

	if !slices.Equal(t.shape, o.shape) {
		if s := midwayBroadcast(o.shape, t.shape); t.Broadable(s...) && !slices.Equal(s, t.shape) {
			t = t.Broadcast(s...)
		}

		if s := midwayBroadcast(t.shape, o.shape); o.Broadable(s...) && !slices.Equal(s, o.shape) {
			o = o.Broadcast(s...)
		}

		if !slices.Equal(t.shape, o.shape) {
			if EnvConfig.Interactive {
				panic(ErrNotBroadable)
			} else {
				t.Err = ErrNotBroadable
				return t
			}
		}
	}

	handleZip(t, o, f, configCPU(t.Numel()))

	return t
}

// Add takes a value and performs elementwise addition
// between other and this Tensor.
func (t Tensor[T]) Add(other any) Tensor[T] {
	return t.Zip(other, func(x, y T) T {
		return x + y
	})
}

// Sub takes a value and performs elementwise subtraction
// between other and this Tensor.
func (t Tensor[T]) Sub(other any) Tensor[T] {
	return t.Zip(other, func(x, y T) T {
		return x - y
	})
}

// Mul takes a value and performs elementwise multiplication
// between other and this Tensor.
func (t Tensor[T]) Mul(other any) Tensor[T] {
	return t.Zip(other, func(x, y T) T {
		return x * y
	})
}

// Div takes a value and performs elementwise division
// between other and this Tensor.
func (t Tensor[T]) Div(other any) Tensor[T] {
	return t.Zip(other, func(x, y T) T {
		return x / y
	})
}
