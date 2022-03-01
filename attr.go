// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"github.com/vorduin/slices"
)

// Iter returns an Iterator over the Tensor's data,
// following the Tensor's indexing scheme.
func (t Tensor[T]) Iter() Iterator[T] {
	return Iterator[T]{
		tensor: t,
		indices: slices.WithLen[int](len(t.shape)),
	}
}

// Ravel returns the Tensor's view in its data buffer.
func (t Tensor[T]) Ravel() []T {
	return t.data[t.offset : t.offset+t.Numel()]
}

// Scalar returns the scalar equivalent of a rank 0 Tensor.
// Panics if the Tensor's rank is not 0.
func (t Tensor[T]) Scalar() T {
	if len(t.shape) != 0 {
		panic("nune: tensor is not rank 0")
	}

	return t.data[t.offset]
}

// Numel returns the number of elements in the Tensor's data buffer.
func (t Tensor[T]) Numel() int {
	if len(t.shape) == 0 {
		return 1
	}

	return int(slices.Prod(t.shape))
}

// Rank returns the Tensor's rank
// (the number of axes in the Tensor's shape).
func (t Tensor[T]) Rank() int {
	return len(t.shape)
}

// Shape returns a copy of the Tensor's shape.
func (t Tensor[T]) Shape() []int {
	return slices.Clone(t.shape)
}

// Stride returns a copy of the Tensor's stride scheme.
func (t Tensor[T]) Stride() []int {
	return slices.Clone(t.stride)
}

// Stride returns the Tensor's view offset in its data buffer.
func (t Tensor[T]) Offset() int {
	return t.offset
}

// Size returns the Tensor's number of dimensions at
// the given axis.
// Panics if axis is out of (0, rank) bounds.
func (t Tensor[T]) Size(axis int) int {
	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		panic(err)
	}

	return t.shape[axis]
}

// Broadable returns whether or not the Tensor can be
// broadcast to the given shape.
func (t *Tensor[T]) Broadable(shape ...int) bool {
	err := verifyGoodShape(shape...)
	if err != nil {
		return false
	}

	err = verifyArgsBounds(len(t.shape), len(shape))
	if err != nil {
		return false
	}

	var s []int

	if len(t.shape) < len(shape) {
		s = slices.WithLen[int](len(shape))
		for i := 0; i < len(shape)-len(t.shape); i++ {
			s[i] = 1
		}
		copy(s[len(shape)-len(t.shape):], t.shape)
	} else {
		s = t.shape
	}

	for i := 0; i < len(shape); i++ {
		if s[i] != shape[i] && s[i] != 1 {
			return false
		}
	}

	return true
}
