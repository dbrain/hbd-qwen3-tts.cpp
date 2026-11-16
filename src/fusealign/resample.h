// Band-limited resampling equal to torchaudio.functional.resample(wav, orig, new) with its defaults
// (sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99).
#pragma once

#include <cstddef>
#include <vector>

namespace fusealign {

std::vector<float> resample(const float * wav, size_t n_samples, int orig_sample_rate, int new_sample_rate);

}  // namespace fusealign
