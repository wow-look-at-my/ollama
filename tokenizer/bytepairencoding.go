package tokenizer

import (
	"fmt"
	"iter"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/dlclark/regexp2"
	"github.com/ollama/ollama/logutil"
)

type BytePairEncoding struct {
	vocab         *Vocabulary
	regexps       []*regexp2.Regexp
	spaceToSpmSep bool
	fragCache     *sync.Map
	pieceCache    *sync.Map
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
		vocab:      vocab,
		fragCache:  &sync.Map{},
		pieceCache: &sync.Map{},
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
	a, b                      int
	rank                      int
	leftID, rightID, mergedID int32
}

type pairHeap []pair

func (h *pairHeap) init() {
	n := len(*h)
	for i := n/2 - 1; i >= 0; i-- {
		h.down(i, n)
	}
}

func (h *pairHeap) push(p pair) {
	*h = append(*h, p)
	h.up(len(*h) - 1)
}

func (h *pairHeap) pop() pair {
	s := *h
	n := len(s) - 1
	s[0], s[n] = s[n], s[0]
	*h = s[:n]
	h.down(0, n)
	return s[n]
}

func (h pairHeap) up(j int) {
	for {
		i := (j - 1) / 2
		if i == j || h[j].rank >= h[i].rank {
			break
		}
		h[i], h[j] = h[j], h[i]
		j = i
	}
}

func (h pairHeap) down(i0, n int) {
	i := i0
	for {
		j := 2*i + 1
		if j >= n || j < 0 {
			break
		}
		if j2 := j + 1; j2 < n && h[j2].rank < h[j].rank {
			j = j2
		}
		if h[j].rank >= h[i].rank {
			break
		}
		h[i], h[j] = h[j], h[i]
		i = j
	}
}

type bpeNode struct {
	p, n int
	id   int32
	r    rune
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

		if cached, ok := bpe.fragCache.Load(frag.value); ok {
			ids = append(ids, cached.([]int32)...)
			continue
		}

		fragStart := len(ids)
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

			if cached, ok := bpe.pieceCache.Load(normalized); ok {
				ids = append(ids, cached.([]int32)...)
				continue
			}

			pieceStart := len(ids)
			runes := []rune(normalized)
			nodes := make([]bpeNode, len(runes))
			for r := range runes {
				nodes[r] = bpeNode{
					p:  r - 1,
					n:  r + 1,
					id: bpe.vocab.Encode(string(runes[r : r+1])),
					r:  runes[r],
				}
			}

			pairwise := func(a, b int) (pair, bool) {
				if a < 0 || b >= len(runes) || nodes[a].id < 0 || nodes[b].id < 0 {
					return pair{}, false
				}
				rank, mergedID, ok := bpe.vocab.MergeByID(nodes[a].id, nodes[b].id)
				if !ok {
					return pair{}, false
				}
				return pair{
					a: a, b: b,
					rank:     rank,
					leftID:   nodes[a].id,
					rightID:  nodes[b].id,
					mergedID: mergedID,
				}, true
			}

			h := make(pairHeap, 0, len(runes))
			for i := range len(runes) - 1 {
				if p, ok := pairwise(i, i+1); ok {
					h = append(h, p)
				}
			}
			h.init()

			for len(h) > 0 {
				p := h.pop()

				if nodes[p.a].id != p.leftID || nodes[p.b].id != p.rightID {
					continue
				}

				nodes[p.a].id = p.mergedID
				nodes[p.b].id = -1

				nodes[p.a].n = nodes[p.b].n
				if nodes[p.b].n < len(nodes) {
					nodes[nodes[p.b].n].p = p.a
				}

				if np, ok := pairwise(nodes[p.a].p, p.a); ok {
					h.push(np)
				}

				if np, ok := pairwise(p.a, nodes[p.a].n); ok {
					h.push(np)
				}
			}

			for idx := 0; idx < len(nodes); idx = nodes[idx].n {
				if nodes[idx].id >= 0 {
					ids = append(ids, nodes[idx].id)
				} else if bpe.spaceToSpmSep {
					for _, b := range []byte(string(nodes[idx].r)) {
						if id := bpe.vocab.Encode(fmt.Sprintf("<0x%02X>", b)); id >= 0 {
							ids = append(ids, id)
						} else {
							slog.Debug("unknown byte token", "byte", b)
						}
					}
				}
			}

			bpe.pieceCache.Store(normalized, slices.Clone(ids[pieceStart:]))
		}

		bpe.fragCache.Store(frag.value, slices.Clone(ids[fragStart:]))
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
