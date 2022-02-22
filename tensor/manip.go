// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Number, V nune.Number](t Tensor[V]) Tensor[T] {
	if t.Err != nil {
		return Tensor[T]{
			Err: t.Err,
		}
	}

	c := slices.WithLen[T](t.Numel())
	for i := 0; i < len(c); i++ {
		c[i] = T(t.data[i])
	}

	return Tensor[T]{
		data:    c,
		shape:   t.shape,
		strides: t.strides,
	}
}

// Copy copies the Tensor and its underlying data.
func (t Tensor[T]) Copy() Tensor[T] {
	if t.Err != nil {
		return Tensor[T]{
			Err: t.Err,
		}
	}

	return Tensor[T]{
		data: slices.Copy(t.data),
		shape: slices.Copy(t.shape),
		strides: slices.Copy(t.strides),
	}
}

// Reshape modifies the Tensor's indexing scheme.
func (t Tensor[T]) Reshape(shape ...int) Tensor[T] {
	if t.Err != nil {
		return Tensor[T]{
			Err: t.Err,
		}
	}

	if len(shape) == 0 && t.Numel() <= 1 {
		return Tensor[T]{
			data: t.data,
		}
	} else {
		err := verifyGoodShape(shape...)
		if err != nil {
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				return Tensor[T]{
					Err: err,
				}
			}
		}

		return Tensor[T]{
			data: t.data,
			shape: slices.Copy(shape),
			strides: configStrides(shape),
		}
	}
}

// Index returns a view over an index of the Tensor.
// Multiple indices can be provided at the same time.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyArgsBounds(len(indices), t.Rank())
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	for i, idx := range indices {
		err = verifyAxisBounds(idx, t.Size(i))
		if err != nil {
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				return Tensor[T]{
					Err: err,
				}
			}
		}
	}

	var offset int

	for i, idx := range indices {
		offset += idx * t.strides[i]
	}

	return Tensor[T]{
		data:    t.data[offset : offset+t.strides[len(indices)-1]],
		shape:   slices.Copy(t.shape[1:]),
		strides: slices.Copy(t.strides[1:]),
	}
}

// Slice returns a view over a slice of the Tensor.
func (t Tensor[T]) Slice(start, end int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyGoodShape(t.shape...) // make sure Tensor rank is not 0
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	err = verifyGoodInterval(start, end, [2]int{0, t.Size(0)})
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	shape := slices.WithLen[int](len(t.shape))
	shape[0] = end - start
	copy(shape[1:], t.shape[1:])

	return Tensor[T]{
		data:    t.data[start*t.strides[0] : end*t.strides[0]],
		shape:   shape,
		strides: slices.Copy(t.strides),
	}
}

func (t Tensor[T]) Broadcast(shape ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	if !t.Broadable(shape...) {
		if nune.EnvConfig.Interactive {
			panic(ErrNotBroadable)
		} else {
			return Tensor[T]{
				Err: ErrNotBroadable,
			}
		}
	}

	var expandedShape []int

	if len(t.shape) < len(shape) {
		expandedShape = slices.WithLen[int](len(shape))
		for i := 0; i < len(shape)-len(t.shape); i++ {
			expandedShape[i] = 1
		}
		copy(expandedShape[len(shape)-len(t.shape):], t.shape)
	} else {
		expandedShape = t.shape
	}

	strides := configStrides(shape)

	data := slices.WithLen[T](int(slices.Prod(shape)))

	var expansion, stride int = 1, strides[0]
	for i := 0; i < len(shape) - 1; i++ {
		if expandedShape[i] != shape[i] {
			expansion *= shape[i]
			stride = strides[i]
		}
	}

	for j := 0; j < expansion; j++ { // specifies which part of the expanded parts
		offset := j * stride 
		for k := 0; k < len(t.data); k++ { // specifies offset in expansion
			expOffset := k * shape[len(shape)-1]
			for l := 0; l < shape[len(shape)-1]; l++ { // specifies index in expansion
				data[offset+expOffset+l] = t.data[k]
			}
		}
	}

	return Tensor[T]{
		data: data,
		shape: slices.Copy(shape),
		strides: strides,
	}
}

// Reverse reverses the order of the elements of the Tensor.
func (t Tensor[T]) Reverse() Tensor[T] {
	if t.Err != nil {
		return t
	}

	for i, j := 0, t.Numel()-1; i < j; i, j = i+1, j-1 {
		t.data[i], t.data[j] = t.data[j], t.data[i]
	}

	return t
}
