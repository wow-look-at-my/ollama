package ollamarunner

import (
	"log/slog"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/tokenizer"
)

const mtpDefaultDraftTokens = 4

func isMTPEligible(m model.Model, seq *Sequence) bool {
	mtpModel, ok := m.(model.MTPModel)
	if !ok || !mtpModel.HasDraft() {
		return false
	}
	if !seq.sampler.IsGreedy() {
		return false
	}
	if seq.logprobs {
		return false
	}
	slog.Debug("MTP eligible, attempting draft cycle")
	return true
}

func runMTPCycle(
	s *Server,
	seq *Sequence,
	token int32,
	logits []float32,
	hiddenFloats []float32,
	hiddenDim int,
	position int32,
	tok tokenizer.Tokenizer,
) (acceptedTokens []int32, nextToken int32, ok bool) {
	mtpModel, valid := s.model.(model.MTPModel)
	if !valid {
		return nil, token, false
	}

	maxDraft := mtpDefaultDraftTokens
	if seq.numPredict > 0 {
		remaining := seq.numPredict - seq.numPredicted
		if remaining <= 1 {
			return nil, token, false
		}
		if maxDraft > remaining-1 {
			maxDraft = remaining - 1
		}
	}

	cache := s.model.Config().Cache
	wc, isWrapper := cache.(*kvcache.WrapperCache)
	seqID := seq.cache.Id

	// Draft phase: speculation tracks phantom cells so they can be rolled back.
	if isWrapper {
		wc.BeginSpeculation(seqID)
	}

	draftCtx := s.model.Backend().NewContext()
	hidden := draftCtx.Input().FromFloats(hiddenFloats, hiddenDim)
	draftTokens, err := mtpModel.MTPDraft(draftCtx, token, hidden, position, seqID, cache, maxDraft)
	draftCtx.Close()
	if err != nil {
		slog.Warn("MTP draft failed", "error", err)
		if isWrapper {
			wc.Rollback()
		}
		return nil, token, false
	}

	if len(draftTokens) == 0 {
		if isWrapper {
			wc.Rollback()
		}
		return nil, token, false
	}

	// Roll back draft's phantom cells, then begin fresh speculation for verify.
	if isWrapper {
		wc.Rollback()
		wc.BeginSpeculation(seqID)
	}

	verifyCtx := s.model.Backend().NewContext()
	accepted, nextAfter, err := mtpModel.MTPVerify(verifyCtx, logits, draftTokens, seqID, position, cache)
	verifyCtx.Close()
	if err != nil {
		slog.Warn("MTP verification failed", "error", err)
		if isWrapper {
			wc.Rollback()
		}
		return nil, token, false
	}

	if isWrapper {
		wc.Commit(accepted)
	}

	if accepted > 0 {
		slog.Debug("MTP accepted", "count", accepted, "total_drafted", len(draftTokens))
	}

	return draftTokens[:accepted], nextAfter, true
}
