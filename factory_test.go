// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"

	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

func TestFull(t *testing.T) {
	tensor := nune.Full(5, []int{2, 2})

	if !slices.Equal(tensor.Ravel(), []int{5, 5, 5, 5}) {
		t.Error("tensor was not initialized with the given value")
	}

	if !slices.Equal(tensor.Shape(), []int{2, 2}) {
		t.Error("tensor was not initialized with the given shape")
	}

	if !slices.Equal(tensor.Stride(), []int{2, 1}) {
		t.Error("tensor was not initialized with the correct stride")
	}

	if tensor.Offset() != 0 {
		t.Error("tensor was not initialized with the correct offset")
	}
}

func TestFullBadShape(t *testing.T) {
	tensor := nune.Full(0, nil)
	if tensor.Err == nil {
		t.Error("tensor was initialized with a nil shape")
	}

	tensor = nune.Full(0, []int{0, 2})
	if tensor.Err == nil {
		t.Error("tensor was initialized with a null axis")
	}

	tensor = nune.Full(0, []int{-1, 2})
	if tensor.Err == nil {
		t.Error("tensor was initialized with a negative axis")
	}
}

func TestFullLike(t *testing.T) {
	tensor := nune.Full(3, []int{2, 2})
	like := nune.FullLike(5, tensor)

	if !slices.Equal(like.Ravel(), []int{5, 5, 5, 5}) {
		t.Error("like was not initialized with the given value")
	}

	if !slices.Equal(like.Shape(), []int{2, 2}) {
		t.Error("like was not initialized with the same shape as tensor")
	}

	if !slices.Equal(like.Stride(), []int{2, 1}) {
		t.Error("like was not initialized with the correct stride")
	}

	if like.Offset() != 0 {
		t.Error("like was not initialized with the correct offset")
	}
}

func TestZeros(t *testing.T) {
	tensor := nune.Zeros[int](2, 2)

	if !slices.Equal(tensor.Ravel(), []int{0, 0, 0, 0}) {
		t.Error("tensor was not initialized with null values")
	}

	if !slices.Equal(tensor.Shape(), []int{2, 2}) {
		t.Error("tensor was not initialized with the correct shape")
	}

	if !slices.Equal(tensor.Stride(), []int{2, 1}) {
		t.Error("tensor was not initialized with the correct stride")
	}

	if tensor.Offset() != 0 {
		t.Error("tensor was not initialized with the correct offset")
	}
}

func TestZerosBadShape(t *testing.T) {
	tensor := nune.Zeros[int](0, 2)
	if tensor.Err == nil {
		t.Error("tensor was initialized with a null axis")
	}

	tensor = nune.Zeros[int](-1, 2)
	if tensor.Err == nil {
		t.Error("tensor was initialized with a negative axis")
	}
}

func TestZerosLike(t *testing.T) {
	tensor := nune.Full(5, []int{2, 2})
	like := nune.ZerosLike[int](tensor)

	if !slices.Equal(like.Ravel(), []int{0, 0, 0, 0}) {
		t.Error("like was not initialized with null values")
	}

	if !slices.Equal(like.Shape(), []int{2, 2}) {
		t.Error("like was not initialized with the same shape as tensor")
	}

	if !slices.Equal(like.Stride(), []int{2, 1}) {
		t.Error("like was not initialized with the correct stride")
	}

	if like.Offset() != 0 {
		t.Error("like was not initialized with the correct offset")
	}
}

func TestOnes(t *testing.T) {
	tensor := nune.Ones[int](2, 2)

	if !slices.Equal(tensor.Ravel(), []int{1, 1, 1, 1}) {
		t.Error("tensor was not initialized with the correct values")
	}

	if !slices.Equal(tensor.Shape(), []int{2, 2}) {
		t.Error("tensor was not initialized with the correct shape")
	}

	if !slices.Equal(tensor.Stride(), []int{2, 1}) {
		t.Error("tensor was not initialized with the correct stride")
	}

	if tensor.Offset() != 0 {
		t.Error("tensor was not initialized with the correct offset")
	}
}

func TestOnesBadShape(t *testing.T) {
	tensor := nune.Ones[int](0, 2)
	if tensor.Err == nil {
		t.Error("tensor was initialized with a null axis")
	}

	tensor = nune.Ones[int](-1, 2)
	if tensor.Err == nil {
		t.Error("tensor was initialized with a negative axis")
	}
}

func TestOnesLike(t *testing.T) {
	tensor := nune.Full(5, []int{2, 2})
	like := nune.OnesLike[int](tensor)

	if !slices.Equal(like.Ravel(), []int{1, 1, 1, 1}) {
		t.Error("like was not initialized with the correct values")
	}

	if !slices.Equal(like.Shape(), []int{2, 2}) {
		t.Error("like was not initialized with the same shape as tensor")
	}

	if !slices.Equal(like.Stride(), []int{2, 1}) {
		t.Error("like was not initialized with the correct stride")
	}

	if like.Offset() != 0 {
		t.Error("like was not initialized with the correct offset")
	}
}

func TestFromBuffer(t *testing.T) {
	buf := []int{1, 2, 3, 4}
	tensor := nune.FromBuffer(buf)

	if !slices.Equal(tensor.Ravel(), buf) {
		t.Error("tensor was not initialized with the correct values")
	}

	if !slices.Equal(tensor.Shape(), []int{4}) {
		t.Error("tensor was not initialized with the same shape as tensor")
	}

	if !slices.Equal(tensor.Stride(), []int{1}) {
		t.Error("tensor was not initialized with the correct stride")
	}

	if tensor.Offset() != 0 {
		t.Error("tensor was not initialized with the correct offset")
	}

	buf[0] = 0
	if !slices.Equal(tensor.Ravel(), buf) {
		t.Error("tensor is not using given buffer as its underlying data buffer")
	}
}

func TestFromBufferNil(t *testing.T) {
	buf := []int{}
	tensor := nune.FromBuffer(buf)

	if tensor.Err == nil {
		t.Error("tensor was initialized using an empty buffer")
	}
}