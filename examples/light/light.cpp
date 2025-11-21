#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// Simple whisper transcription with minimal parameters
// - Language: English
// - CPU only (no GPU)
// - Single thread
// - No VAD (Voice Activity Detection)
// - Basic text output only

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    // Check command line arguments
    if (argc != 2) {
        fprintf(stderr, "usage: %s <audio_file>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "A simple transcription tool that transcribes audio to text.\n");
        fprintf(stderr, "Fixed settings: English, CPU only, single thread, no VAD\n");
        fprintf(stderr, "Supported audio formats: flac, mp3, ogg, wav\n");
        return 1;
    }

    const char * fname_inp = argv[1];
    const char * model_path = "models/ggml-base.en.bin";

    // Check if input file exists
    if (!is_file_exist(fname_inp)) {
        fprintf(stderr, "error: input file not found '%s'\n", fname_inp);
        return 2;
    }

    // Set up parameters - single thread only
    const int n_threads = 1;

    // Initialize whisper context with CPU-only settings
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;     // CPU only
    cparams.flash_attn = false;  // No flash attention (GPU feature)

    struct whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        fprintf(stderr, "make sure the model file exists at: %s\n", model_path);
        return 3;
    }

    // Read audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;

    if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp);
        whisper_free(ctx);
        return 4;
    }

    // Print processing info
    fprintf(stderr, "processing '%s' (%d samples, %.1f sec), %d thread\n",
            fname_inp, int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE, n_threads);

    // Set up whisper parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.n_threads        = n_threads;
    wparams.no_timestamps    = false;
    wparams.token_timestamps = false;
    wparams.temperature      = 0.0f;
    wparams.temperature_inc  = 0.0f;

    // Capture top 20 candidate tokens with their logprobs
    wparams.capture_top_candidates = true;
    wparams.n_top_candidates       = 20;

    // No VAD
    wparams.vad = false;

    // Run the transcription
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stderr, "error: failed to process audio\n");
        whisper_free(ctx);
        return 5;
    }

    // Print all segments after processing completes
    printf("\n");
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        const char * text = whisper_full_get_segment_text(ctx, i);

        printf("[%s --> %s]  %s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);

        // Print top candidates for each token
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; j++) {
            const char * token_text = whisper_full_get_token_text(ctx, i, j);
            const float token_p = whisper_full_get_token_p(ctx, i, j);

            printf("  Token %d: '%s' (p=%.4f)\n", j, token_text, token_p);

            // Print top candidates
            const int n_candidates = whisper_full_n_top_candidates(ctx, i, j);
            if (n_candidates > 0) {
                printf("    Top %d candidates:\n", n_candidates);
                for (int k = 0; k < n_candidates; k++) {
                    whisper_token_candidate cand = whisper_full_get_top_candidate(ctx, i, j, k);
                    const char * cand_text = whisper_token_to_str(ctx, cand.id);
                    printf("      %2d. '%s' (p=%.4f, logp=%.4f)\n",
                           k + 1, cand_text, cand.p, cand.plog);
                }
            }
        }
        printf("\n");
    }

    // Clean up
    whisper_free(ctx);

    return 0;
}
