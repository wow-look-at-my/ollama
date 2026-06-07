package version

import "runtime/debug"

// Version is the ollama version.
//
// Release builds set it to the release tag via -ldflags "-X" (see
// scripts/push_docker.sh / .github/workflows/release.yaml). For every other
// build it is left at the placeholder below and then replaced, in init, with the
// VCS revision Go automatically embeds in the binary's build info. That keeps
// source/dev/CI builds reporting the real commit instead of 0.0.0 without an -X
// ldflag — which would force a relink on every commit and defeat build caching.
var Version = "0.0.0"

func init() {
	// A non-placeholder value means a release build stamped it via -ldflags;
	// leave that (a real semver) alone.
	if Version != "0.0.0" {
		return
	}
	if v, ok := vcsVersion(debug.ReadBuildInfo()); ok {
		Version = v
	}
}

// vcsVersion derives a version string from the VCS metadata Go records in a
// binary's build info: the short commit revision, suffixed with "-dirty" when
// the working tree had uncommitted changes at build time. It takes the result of
// debug.ReadBuildInfo directly and returns ok=false when there is nothing usable
// (no build info, or no recorded revision), so the caller keeps the placeholder.
func vcsVersion(info *debug.BuildInfo, ok bool) (string, bool) {
	if !ok || info == nil {
		return "", false
	}

	var revision string
	var modified bool
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}

	if revision == "" {
		return "", false
	}

	if len(revision) > 12 {
		revision = revision[:12]
	}
	if modified {
		revision += "-dirty"
	}
	return revision, true
}
