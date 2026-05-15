package tokenizer

import (
	"log/slog"
	"slices"
	"strings"
	"sync"
)

type Special int32

const (
	SpecialBOS Special = iota
	SpecialEOS
)

type Vocabulary struct {
	Values []string
	Types  []int32
	Scores []float32
	Merges []string

	BOS, EOS       []int32
	AddBOS, AddEOS bool

	specialOnce sync.Once
	special     []string

	valuesOnce sync.Once
	values     map[string]int32

	mergeOnce sync.Once
	merge     map[string]int32

	mergeByIDOnce sync.Once
	mergeByID     map[uint64]int32
}

func (v *Vocabulary) Is(id int32, special Special) bool {
	switch special {
	case SpecialBOS:
		return slices.Contains(v.BOS, id)
	case SpecialEOS:
		return slices.Contains(v.EOS, id)
	default:
		return false
	}
}

func (v *Vocabulary) addSpecials(ids []int32) []int32 {
	if v.AddBOS && len(v.BOS) > 0 {
		if len(ids) > 0 && slices.Contains(v.BOS, ids[0]) {
			slog.Warn("adding bos token to prompt which already has it", "id", v.BOS)
		}

		slog.Debug("adding bos token to prompt", "id", v.BOS[0])
		ids = append([]int32{v.BOS[0]}, ids...)
	}

	if v.AddEOS && len(v.EOS) > 0 {
		if len(ids) > 0 && slices.Contains(v.BOS, ids[len(ids)-1]) {
			slog.Warn("adding eos token to prompt which already has it", "id", v.EOS)
		}

		slog.Debug("adding eos token to prompt", "id", v.EOS[0])
		ids = append(ids, v.EOS[0])
	}

	return ids
}

func (v *Vocabulary) Encode(s string) int32 {
	v.valuesOnce.Do(func() {
		v.values = make(map[string]int32, len(v.Values))
		for i, value := range v.Values {
			v.values[value] = int32(i)
		}
	})

	if id, ok := v.values[s]; ok {
		return id
	}

	return -1
}

func (v *Vocabulary) Decode(id int32) string {
	return v.Values[id]
}

func (v *Vocabulary) SpecialVocabulary() []string {
	v.specialOnce.Do(func() {
		for i := range v.Values {
			if v.Types[i] == TOKEN_TYPE_CONTROL || v.Types[i] == TOKEN_TYPE_USER_DEFINED {
				v.special = append(v.special, v.Values[i])
			}
		}
	})

	return v.special
}

func (v *Vocabulary) Merge(left, right string) int {
	v.mergeOnce.Do(func() {
		v.merge = make(map[string]int32, len(v.Merges))
		for i, merge := range v.Merges {
			v.merge[merge] = int32(i)
		}
	})

	if id, ok := v.merge[left+" "+right]; ok {
		return int(id)
	}

	return -1
}

func mergePairKey(id1, id2 int32) uint64 {
	return uint64(uint32(id1))<<32 | uint64(uint32(id2))
}

func (v *Vocabulary) MergeByID(left, right int32) int {
	v.mergeByIDOnce.Do(func() {
		v.mergeByID = make(map[uint64]int32, len(v.Merges))
		for i, m := range v.Merges {
			parts := strings.SplitN(m, " ", 2)
			if len(parts) != 2 {
				continue
			}
			leftID := v.Encode(parts[0])
			rightID := v.Encode(parts[1])
			if leftID >= 0 && rightID >= 0 {
				v.mergeByID[mergePairKey(leftID, rightID)] = int32(i)
			}
		}
	})

	if id, ok := v.mergeByID[mergePairKey(left, right)]; ok {
		return int(id)
	}

	return -1
}
