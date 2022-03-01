# Nune

A tensor based numerical engine.

The numerical engine, or nune for short, is a package for performing numerical computation in Go, relying on generic tensors.
This package provides facilities to manipulate and perform various operations on numerical data, and implements a numeric n-dimensional generic Tensor, along with a set of functions to create, manipulate and operate on that Tensor.

## Table of contents

- [Installation](#Installation)
- [Philosophy](#Philosophy)
- [Design](#Design)
- [Performance](#Performance)
- [Usage](#Usage)
- [Roadmap](#Roadmap)
- [License](#License)

## Installation

Nune requires Go v1.18 as it's entirely based on generics in order to achieve a flexible interface.
Go v1.18 is currently only available in beta version, which can be downloaded [here](https://go.dev/dl/).

After installing Go1.18, simply run this in your terminal...

```zsh
go get github.com/vorduin/nune
```

... and you're good to Go!

## Philosophy

Nune follows Go's principles and design philosophies of simplicity and minimalism.
Therefore, going forward, Nune will always be a compact library providing only the minimal and foundational functions to deal with numerical data and its computation.
Furthermore, this means Nune's main focus will be Go, meaning it's main computation target is CPU computing. This leads to a cleaner API optimized to do what Go was created for; perform high-speed computation on large scale distributed systems.

## Design

Nune tries to expose a clear, minimal, yet expressive API. This has two consequences; how does it represent the data in memory, and how does it's API deal with that data.

First off, Nune represents all tensor data densely, as a contiguous 1-dimensional buffer in memory. In order to keep a minimum memory overhead, all tensor operations that do not alter the data return views over the tensor's data buffer. For example, indexing, reshaping or permutating a tensor simply alter the tensor's indexing scheme over the same data buffer, without making any copies. This both saves memory and provies a really efficient way to manipulate the tensor.

Next up is how Nune performs operations on a tensor. The simplest and most efficient design was to provide all operations as methods of the tensor, all of which are computed inplace. In case a certain operation shouldn't affect the original data, a simple `Clone` method can be called, returning a completely identical tensor, with a copy of the underlying data. All tensor operations are written to work in a functional style.

## Performance

Nune is a high performance numerical engine. However, high performance is subject to a lot of factors, many of which differ from machine to machine.
Therefore, a number of benchmarks have been written and can be run using the `go test` command, where Nune's performance can then be compared to that of other libraries on that particular machine.

However, to provide an idea of Nune's capabilities, here is a table showcasing Nune's performances compared to NumPy's and PyTorch's on the same machine, with Nune running in single-threaded mode (couldn't find out if NumPy and PyTorch are running single-threaded though), all operating on a Tensor with 1e7 elements with a type of float64:

|      | Nune | NumPy | PyTorch |
|------|------|-------|---------|
| Min  |  3ms | 5ms   | 5ms     |
| Prod | 3ms  | 10ms  | 5ms     |
| Div  | 45ms | 15ms  | 10ms    |
| Abs  | 10ms | 25ms  | 20ms    |
| Sqrt | 12ms | 40ms  | 20ms    |
| Tanh | 50ms | 55ms  | 65ms    |

As can be concluded, Nune is exceptionally fast on pointwise and reduction operations, but is still somewhat slower on operations that are SIMD accelerated by its counterparts. This is of course to be improved in the future, as the library is still not optimized for maximum performance on all operations.

## Usage

```Go
package main

import (
	"fmt"
	"math"

	"github.com/vorduin/nune"
)

func main() {
	// Nune can use any numeric type's superset
	type Freq float64

	// or ...strings...
	b := nune.From[byte]("nune's moon")
	nune.FmtConfig.Btoa = true // convert bytes to ASCII
	nune.FmtConfig.Excerpt = 12 // max num of elements formatted in an axis
	fmt.Println(b)
	// Prints:
	//
	// Tensor([n, u, n, e, ', s,  , m, o, o, n])

	// Create a rank 1 Tensor from the range (0, 10)
	t := nune.Range[Freq](0, 10, 1)

	// Create 10 copies of the Tensor's data, concatenate them,
	// and flip axis 1
	t = t.Repeat(10).Flip(1) // shape is now (10, 10)

	// Operations between two tensors automatically
	// broadcast the tensor's together
	// here the two tensors' shapes are (4, 25, 1) and (4),
	// so the resulting shape ends up being (4, 25, 4)
	_ = t.Clone().Reshape(4, 25, 1).Add([]int{1, 2, 3, 4}).Permute(1, 0, 2)

	// Nune is designed to work for both libraries
	res := t.Reshape(10)
	if res.Err != nil {
		panic("i can see the moon")
	}

	// or interactively, such as working in
	// a notebooks environment
	nune.EnvConfig.Interactive = true

	// The following line automatically panics
	// res = t.Reshape(10)

	// Nune allows you to define custom functions
	// any way you want
	//
	// The following is a pointwise sigmoid function
	// that can be parallelized by nune's backend
	res = t.Map(func(x Freq) Freq {
		return Freq(1 / (1 + math.Exp(-float64(x))))
	})

	// or you could use nune's functional API
	res = t.Copysign(-1).Exp().Add(1).Pow(-1)
	// in the above chain, if nune is running
	// in a non-interactive environment and
	// one operation fails, all subsequent
	// operations would fail and would return
	// the original corresponding error.

	nune.FmtConfig.Excerpt = 4
	nune.FmtConfig.Precision = 2 // num of decimals formatted

	fmt.Println(res)
	// Prints:
	//
	// Tensor([[0.73, 0.73, ..., 0.68, 0.62]
	//         [0.73, 0.73, ..., 0.68, 0.62]
	//         ...,
	//         [0.73, 0.73, ..., 0.68, 0.62]
	//         [0.73, 0.73, ..., 0.68, 0.62]])

	// and much more!
	// Nune impelements most of the "math" package
	// as Tensor methods, many tensor manipulation functions,
	// and facilities to facilitate and speed up working
	// with and operating on numerical data.
}
```

## Roadmap

Because Nune's philosophy is to provide only the minimal, foundational numerical facilities, its roadmap isn't so far ahead, and Nune itself is already close to reaching a stable state.
Limitations that need to be fixed before this is a rock-stable library are the following, in order:

- Rewriting the underlying API to be simple and clean, with direct algorithms and clear error messages.
- Optimizing the backend for maximum performance.
- Rigorously testing and benchmarking the API and its underlying algorithms.
- Stabilizing the API for forward compatiblity, and making sure none of the current decisions will be regretted in the future because of backward compatiblity.
- Writing examples to ease the use of this library.

## License

Nune has a BSD-style license, as found in the [LICENSE](https://github.com/vorduin/nune/blob/main/LICENSE) file.
