package tokenizer

import (
	"container/heap"
	"fmt"
	"iter"
	"log/slog"
	"slices"
	"strconv"
	"strings"

	"github.com/dlclark/regexp2"
	"github.com/ollama/ollama/logutil"
)

type BytePairEncoding struct {
	vocab         *Vocabulary
	regexps       []*regexp2.Regexp
	spaceToSpmSep bool // When true, normalize spaces to ▁ instead of GPT-2 byte-level encoding
}

var _ Tokenizer = (*BytePairEncoding)(nil)

// BPEOption configures BytePairEncoding behavior
type BPEOption func(*BytePairEncoding)

// WithSentencePieceNormalizer enables ▁ space normalization instead of GPT-2 byte-level encoding.
func WithSentencePieceNormalizer() BPEOption {
	return func(bpe *BytePairEncoding) {
		bpe.spaceToSpmSep = true
	}
}

func NewBytePairEncoding(vocab *Vocabulary, pretokenizer ...string) BytePairEncoding {
	return newBytePairEncoding(vocab, pretokenizer)
}

func NewBytePairEncodingWithOptions(vocab *Vocabulary, pretokenizer []string, opts ...BPEOption) BytePairEncoding {
	bpe := newBytePairEncoding(vocab, pretokenizer, opts...)
	return bpe
}

func newBytePairEncoding(vocab *Vocabulary, pretokenizer []string, opts ...BPEOption) BytePairEncoding {
	bpe := BytePairEncoding{
		vocab: vocab,
	}

	for _, opt := range opts {
		opt(&bpe)
	}

	if len(pretokenizer) == 0 && !bpe.spaceToSpmSep {
		// set default byte-level pretokenizer if none provided, e.g.
		// https://github.com/huggingface/tokenizer/blob/main/tokenizer/src/pre_tokenizer/byte_level.rs#L44
		pretokenizer = []string{`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`}
	}

	bpe.regexps = slices.Collect(func(yield func(*regexp2.Regexp) bool) {
		for _, p := range pretokenizer {
			if !yield(regexp2.MustCompile(p, regexp2.RE2)) {
				return
			}
		}
	})

	return bpe
}

func (bpe BytePairEncoding) Vocabulary() *Vocabulary {
	return bpe.vocab
}

func (bpe BytePairEncoding) Is(id int32, special Special) bool {
	return bpe.vocab.Is(id, special)
}

func (bpe *BytePairEncoding) split(s string) iter.Seq[string] {
	parts := []string{s}
	for _, re := range bpe.regexps {
		parts = slices.Collect(func(yield func(string) bool) {
			for _, part := range parts {
				r := []rune(part)
				var offset int
				for m, _ := re.FindRunesMatch(r); m != nil; m, _ = re.FindNextMatch(m) {
					if offset-m.Index != 0 {
						if !yield(string(r[offset:m.Index])) {
							return
						}
					}

					if !yield(m.String()) {
						return
					}

					offset = m.Index + m.Length
				}

				if offset < len(r) {
					if !yield(string(r[offset:])) {
						return
					}
				}
			}
		})
	}

	return slices.Values(parts)
}

// fragment is a string fragment and their corresponding token IDs
type fragment struct {
	value string
	ids   []int32
}

type pair struct {
	a, b            int
	rank            int
	leftID, rightID int32
}

type pairHeap []*pair

func (h pairHeap) Len() int            { return len(h) }
func (h pairHeap) Less(i, j int) bool  { return h[i].rank < h[j].rank }
func (h pairHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(x any)         { *h = append(*h, x.(*pair)) }
func (h *pairHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return x
}

type merge struct {
	p, n  int
	runes []rune
	id    int32
}

func (bpe BytePairEncoding) Encode(s string, addSpecial bool) ([]int32, error) {
	fragments := []fragment{{value: s}}
	for _, special := range bpe.vocab.SpecialVocabulary() {
		// TODO: process special tokens concurrently
		id := bpe.vocab.Encode(special)
		for i := 0; i < len(fragments); i++ {
			frag := fragments[i]
			if len(frag.ids) > 0 {
				continue
			}

			var middle []fragment
			switch i := strings.Index(frag.value, special); {
			case i < 0:
				middle = append(middle, frag)
			case i > 0:
				middle = append(middle, fragment{value: frag.value[:i]})
				fallthrough
			default:
				middle = append(middle, fragment{value: special, ids: []int32{id}})
				if rest := frag.value[i+len(special):]; rest != "" {
					middle = append(middle, fragment{value: rest})
				}
			}

			fragments = append(fragments[:i], append(middle, fragments[i+1:]...)...)
		}
	}

	var ids []int32
	for _, frag := range fragments {
		if len(frag.ids) > 0 {
			ids = append(ids, frag.ids...)
			continue
		}

		for split := range bpe.split(frag.value) {
			// TODO: process splits concurrently
			var normalized string
			if bpe.spaceToSpmSep {
				// SentencePiece-style: replace spaces with ▁
				normalized = strings.ReplaceAll(split, " ", spmWhitespaceSep)
			} else {
				// GPT-2 byte-level: map bytes to shifted Unicode codepoints
				var sb strings.Builder
				for _, b := range []byte(split) {
					r := rune(b)
					switch {
					case r == 0x00ad:
						r = 0x0143
					case r <= 0x0020:
						r = r + 0x0100
					case r >= 0x007f && r <= 0x00a0:
						r = r + 0x00a2
					}
					sb.WriteRune(r)
				}
				normalized = sb.String()
			}

			// short circuit if the fragment is in the vocabulary
			if id := bpe.vocab.Encode(normalized); id >= 0 {
				ids = append(ids, id)
				continue
			}

			runes := []rune(normalized)
			nodes := make([]merge, len(runes))
			for r := range runes {
				nodes[r] = merge{
					p:     r - 1,
					n:     r + 1,
					runes: []rune{runes[r]},
					id:    bpe.vocab.Encode(string(runes[r : r+1])),
				}
			}

			pairwise := func(a, b int) *pair {
				if a < 0 || b >= len(runes) {
					return nil
				}

				rank := bpe.vocab.MergeByID(nodes[a].id, nodes[b].id)
				if rank < 0 {
					return nil
				}

				return &pair{
					a:       a,
					b:       b,
					rank:    rank,
					leftID:  nodes[a].id,
					rightID: nodes[b].id,
				}
			}

			h := &pairHeap{}
			for i := range len(runes) - 1 {
				if p := pairwise(i, i+1); p != nil {
					*h = append(*h, p)
				}
			}
			heap.Init(h)

			for h.Len() > 0 {
				p := heap.Pop(h).(*pair)

				left, right := nodes[p.a], nodes[p.b]
				if left.id != p.leftID || right.id != p.rightID {
					continue
				}

				merged := string(left.runes) + string(right.runes)
				mergedID := bpe.vocab.Encode(merged)
				if mergedID < 0 {
					continue
				}

				nodes[p.a].runes = append(left.runes, right.runes...)
				nodes[p.a].id = mergedID
				nodes[p.b].runes = nil
				nodes[p.b].id = -1

				nodes[p.a].n = right.n
				if right.n < len(nodes) {
					nodes[right.n].p = p.a
				}

				if np := pairwise(nodes[p.a].p, p.a); np != nil {
					heap.Push(h, np)
				}

				if np := pairwise(p.a, nodes[p.a].n); np != nil {
					heap.Push(h, np)
				}
			}

			for _, merge := range nodes {
				if len(merge.runes) > 0 {
					if id := bpe.vocab.Encode(string(merge.runes)); id >= 0 {
						ids = append(ids, id)
					} else if bpe.spaceToSpmSep {
						// SentencePiece byte fallback: encode each UTF-8 byte as <0xHH>
						for _, b := range []byte(string(merge.runes)) {
							if id := bpe.vocab.Encode(fmt.Sprintf("<0x%02X>", b)); id >= 0 {
								ids = append(ids, id)
							} else {
								slog.Debug("unknown byte token", "byte", b)
							}
						}
					}
				}
			}
		}
	}

	if addSpecial {
		ids = bpe.vocab.addSpecials(ids)
	}

	logutil.Trace("encoded", "string", s, "ids", ids)
	return ids, nil
}

type lazyIdsString struct {
	ids []int32
}

func (l lazyIdsString) LogValue() slog.Value {
	return slog.AnyValue(fmt.Sprint(l.ids))
}

func (bpe BytePairEncoding) Decode(ids []int32) (string, error) {
	var sb strings.Builder

	// SentencePiece-style BPE stores true Unicode codepoints in the vocab
	// (plus ▁ as a whitespace marker), so decoding should pass runes through
	// directly instead of applying the GPT-2 byte-level reverse mapping.
	// Without this, codepoints in the 0x0100-0x0142 range (e.g. ą ę ć ł)
	// get mangled by the GPT-2 reversal into control characters.
	if bpe.spaceToSpmSep {
		for _, id := range ids {
			data := bpe.vocab.Decode(id)

			// SentencePiece byte tokens: "<0xHH>" → raw byte
			if len(data) == 6 && strings.HasPrefix(data, "<0x") && strings.HasSuffix(data, ">") {
				if b, err := strconv.ParseUint(data[3:5], 16, 8); err == nil {
					sb.WriteByte(byte(b))
					continue
				}
			}

			for _, r := range data {
				if r == 0x2581 { // ▁ (LOWER ONE EIGHTH BLOCK)
					sb.WriteByte(' ')
				} else {
					sb.WriteRune(r)
				}
			}
		}

		logutil.Trace("decoded", "string", sb.String(), "from", lazyIdsString{ids: ids})
		return sb.String(), nil
	}

	for _, id := range ids {
		for _, r := range bpe.vocab.Decode(id) {
			// GPT-2 byte-level BPE uses Unicode chars in the 0x0100-0x0143
			// range to represent bytes. Remap them back to actual bytes.
			switch {
			case r == 0x0100:
				// this produces 0x00 aka NULL
				continue
			case r == 0x0143:
				r = 0x00ad
			case r > 0x0100 && r <= 0x0120:
				r = r - 0x0100
			case r > 0x0120 && r <= 0x0142:
				r = r - 0x00a2
			case r > 0x0143:
				// Non-GPT2 rune (e.g., SentencePiece-style BPE).
				// Handle ▁ as word separator, otherwise write the rune as-is.
				if r == 0x2581 { // ▁ (LOWER ONE EIGHTH BLOCK)
					sb.WriteByte(' ')
				} else {
					sb.WriteRune(r)
				}
				continue
			}

			// NOTE: not using WriteRune here because it writes the UTF-8
			// encoding of the rune which is _not_ what we want
			if err := sb.WriteByte(byte(r)); err != nil {
				return "", err
			}
		}
	}

	logutil.Trace("decoded", "string", sb.String(), "from", lazyIdsString{ids: ids})
	return sb.String(), nil
}
