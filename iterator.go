// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"fmt"
	"strings"

	"github.com/vorduin/slices"
)

// An Iterator iterates over a Tensor's data according
// to its corresponding layout.
type Iterator[T Number] struct {
	tensor Tensor[T]
	indices []int
}

// Next increments the Iterator's indices and returns
// the corresponding Tensor element according to the Iterator's
// indices, with a value of true if it exists, and a nil value
// with a value of false if it doesn't.
func (it Iterator[T]) Next() (Tensor[T], bool) {
	if it.tensor.Err != nil {
		if EnvConfig.Interactive {
			panic(it.tensor.Err)
		} else {
			return Tensor[T]{
				Err: it.tensor.Err,
			}, false
		}
	}

	shape := it.tensor.shape

	if !slices.Equal(it.indices, shape) {
		defer func() {
			for i := len(it.indices)-1; i >= 0; i-- {
				if it.indices[i] < shape[i]-1 {
					it.indices[i]++
					for j := i+1; j < len(it.indices); j++ {
						it.indices[j] = 0
					}
					break
				}
				continue
			}
		}()
		
		return it.tensor.Index(it.indices...), true
	}
	return Tensor[T]{}, false
}

// Size returns the Iterator's total number of elements.
func (it Iterator[t]) Size() int {
	return it.tensor.Numel()
}

// Reset resets the Iterator's indices.
func (it Iterator[t]) Reset() {
	for i := range it.indices {
		it.indices[i] = 0
	}
}

// String returns a string representation of the Iterator.
func (it Iterator[T]) String() string {
	if it.tensor.Err != nil {
		if EnvConfig.Interactive {
			panic(it.tensor.Err)
		} else {
			return "Iterator(error)"
		}
	}

	indices := slices.Clone(it.indices)
	it.Reset()

	template := "Iterator({})"
	f := newFmtState(template, it.tensor)

	s := strings.Replace(template, "{}", fmtIterator(it, f), 1)

	it.indices = indices

	return fmt.Sprintf("%s", s)
}

// fmtIterator formats the Iterator into a string.
func fmtIterator[T Number](it Iterator[T], s fmtState) string {
	var b strings.Builder

	b.WriteString("[")

	if it.tensor.Numel() > FmtConfig.Excerpt*2 {
		b.WriteString(it.fmtExcerpted(s))
	} else {
		b.WriteString(it.fmtComplete(s))
	}

	b.WriteString("]")

	return b.String()
}

// fmtExcerpted formats an excerpted representation of
// the Iterator into a string.

func (it Iterator[T]) fmtExcerpted(s fmtState) string {
	var b strings.Builder

	size := it.Size()
	var idx int
	for i := 0; i < size; i++ {
		v, _ := it.Next()

		if idx == FmtConfig.Excerpt-1 {
			b.WriteString(fmtNum(T(v.Scalar()), s))
			b.WriteString(", ..., ")
		} else if idx < FmtConfig.Excerpt || idx >= size - FmtConfig.Excerpt {
			b.WriteString(fmtNum(v.Scalar(), s))

			if idx < FmtConfig.Excerpt-1 || idx < size-1  {
				b.WriteString(", ")
			}
		}
		
		idx++
	}

	return b.String()
}

// fmtComplete formats a complete representation of
// the Iterator into a string.
func (it Iterator[T]) fmtComplete(s fmtState) string {
	var b strings.Builder

	size := it.Size()
	var idx int
	for i := 0; i < size; i++ {
		v, _ := it.Next()

		b.WriteString(fmtNum(v.Scalar(), s))

		if idx < size-1 {
			b.WriteString(", ")
		}
		idx++
	}

	return b.String()
}