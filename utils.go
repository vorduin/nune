// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"math"
	"runtime"
	"github.com/vorduin/slices"
)

// configStride returns the corresponding stride to the given shape.
func configStride(shape []int) []int {
	if len(shape) != 0 {
		stride := slices.WithLen[int](len(shape))
		if len(shape) == 1 {
			stride[0] = 1
		} else {
			stride[0] = slices.Prod(shape[1:])
		}
		
		for i := 1; i < len(shape); i++ {
			stride[i] = stride[i-1] / shape[i]
		}

		return stride
	}

	return nil
}

// configCPU returns the number of CPU cores to use 
// depending on the data's size.
func configCPU(size int) int {
	if EnvConfig.NumCPU != 0 {
		return int(math.Min(float64(size), float64(EnvConfig.NumCPU)))
	}

	bias := float64(size) / 4096 // this is handcoded, therefore beautiful. or ugly

	if bias < 1 {
		return 1
	} else if bias < float64(runtime.NumCPU()) {
		return int(math.Round(bias))
	} else {
		return runtime.NumCPU()
	}
}