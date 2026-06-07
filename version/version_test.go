package version

import (
	"runtime/debug"
	"testing"
)

func TestVCSVersion(t *testing.T) {
	rev := func(revision, modified string) *debug.BuildInfo {
		var settings []debug.BuildSetting
		if revision != "" {
			settings = append(settings, debug.BuildSetting{Key: "vcs.revision", Value: revision})
		}
		if modified != "" {
			settings = append(settings, debug.BuildSetting{Key: "vcs.modified", Value: modified})
		}
		return &debug.BuildInfo{Settings: settings}
	}

	cases := []struct {
		name    string
		info    *debug.BuildInfo
		ok      bool
		wantStr string
		wantOK  bool
	}{
		{name: "no build info", info: nil, ok: false, wantOK: false},
		{name: "nil info but ok", info: nil, ok: true, wantOK: false},
		{name: "no revision recorded", info: rev("", "false"), ok: true, wantOK: false},
		{name: "clean short revision", info: rev("abcdef0123456789", "false"), ok: true, wantStr: "abcdef012345", wantOK: true},
		{name: "dirty revision", info: rev("abcdef0123456789", "true"), ok: true, wantStr: "abcdef012345-dirty", wantOK: true},
		{name: "short revision left intact", info: rev("abc123", "false"), ok: true, wantStr: "abc123", wantOK: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := vcsVersion(tc.info, tc.ok)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if got != tc.wantStr {
				t.Fatalf("version = %q, want %q", got, tc.wantStr)
			}
		})
	}
}
