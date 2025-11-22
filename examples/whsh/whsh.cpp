#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <termios.h>
#include <unistd.h>

// Interactive whisper shell (whsh)
// Usage: whsh <audio_file>
// After transcription, enters interactive mode with commands:
// - help, ? : Show available commands
// - pos N top [K] : Show top K candidate tokens at position N (default K=10)
// - pos N id TID : Force token TID at position N and re-transcribe from that point
// - quit, exit : Exit the shell
// - Arrow Up/Down : Navigate command history
//
// Architecture: Encode-once, decode-many
// - Audio is encoded once at startup (populates kv_cross cache)
// - Initial transcription uses whisper_full (reuses encoded kv_cross)
// - Re-transcriptions use decode-only loop (reuses same kv_cross)

// Structure to map global token position to segment/token indices
struct TokenPosition {
    int segment_idx;
    int token_idx;
};

// Structure for token with top candidates (for decode-only results)
struct DecodedToken {
    whisper_token id;
    float prob;
    std::vector<std::pair<whisper_token, float>> top_candidates;  // id, prob pairs
};

// Structure for decode-only transcription results
struct DecodeResult {
    std::vector<DecodedToken> tokens;
    bool success;
};

// Softmax helper
static void softmax(float * x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// Decode-only transcription (reuses kv_cross from prior encoding)
// This function does NOT re-encode - it assumes kv_cross is already populated
static DecodeResult decode_only(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & forced_tokens,  // tokens to force at the start
    int n_top_candidates = 20,
    int max_tokens = 224  // max tokens to generate
) {
    DecodeResult result;
    result.success = false;

    const int n_vocab = whisper_n_vocab(ctx);
    const int n_threads = 1;

    // Special tokens
    const whisper_token token_sot = whisper_token_sot(ctx);
    const whisper_token token_eot = whisper_token_eot(ctx);
    const whisper_token token_transcribe = whisper_token_transcribe(ctx);
    const whisper_token token_no_timestamps = whisper_token_not(ctx);
    const whisper_token token_lang = whisper_token_lang(ctx, whisper_lang_id("en"));

    // Build initial prompt: [sot, lang, transcribe, no_timestamps]
    std::vector<whisper_token> prompt;
    prompt.push_back(token_sot);
    prompt.push_back(token_lang);
    prompt.push_back(token_transcribe);
    prompt.push_back(token_no_timestamps);

    // Decode the initial prompt
    if (whisper_decode(ctx, prompt.data(), prompt.size(), 0, n_threads) != 0) {
        fprintf(stderr, "decode_only: failed to decode initial prompt\n");
        return result;
    }

    int n_past = prompt.size();

    // Process forced tokens first (if any)
    for (size_t i = 0; i < forced_tokens.size(); i++) {
        whisper_token tok = forced_tokens[i];

        // Decode this token
        if (whisper_decode(ctx, &tok, 1, n_past, n_threads) != 0) {
            fprintf(stderr, "decode_only: failed to decode forced token %d\n", (int)i);
            return result;
        }

        // Get logits and compute probabilities for top candidates
        float * logits = whisper_get_logits(ctx);
        std::vector<float> probs(n_vocab);
        memcpy(probs.data(), logits, n_vocab * sizeof(float));
        softmax(probs.data(), n_vocab);

        // Store this token with its probability and top candidates
        DecodedToken decoded;
        decoded.id = tok;
        decoded.prob = probs[tok];

        // Get top candidates
        std::vector<std::pair<float, whisper_token>> candidates;
        for (int v = 0; v < n_vocab; v++) {
            candidates.push_back({probs[v], v});
        }
        std::partial_sort(candidates.begin(), candidates.begin() + n_top_candidates,
                         candidates.end(), std::greater<std::pair<float, whisper_token>>());

        for (int k = 0; k < n_top_candidates; k++) {
            decoded.top_candidates.push_back({candidates[k].second, candidates[k].first});
        }

        result.tokens.push_back(decoded);
        n_past++;

        if (tok == token_eot) {
            result.success = true;
            return result;
        }
    }

    // Continue with greedy sampling until EOT or max tokens
    while ((int)result.tokens.size() < max_tokens) {
        // Get logits from previous decode
        float * logits = whisper_get_logits(ctx);

        // Compute probabilities
        std::vector<float> probs(n_vocab);
        memcpy(probs.data(), logits, n_vocab * sizeof(float));
        softmax(probs.data(), n_vocab);

        // Find best token (greedy)
        whisper_token best_token = 0;
        float best_prob = probs[0];
        for (int v = 1; v < n_vocab; v++) {
            if (probs[v] > best_prob) {
                best_prob = probs[v];
                best_token = v;
            }
        }

        // Store this token with top candidates
        DecodedToken decoded;
        decoded.id = best_token;
        decoded.prob = best_prob;

        // Get top candidates
        std::vector<std::pair<float, whisper_token>> candidates;
        for (int v = 0; v < n_vocab; v++) {
            candidates.push_back({probs[v], v});
        }
        std::partial_sort(candidates.begin(), candidates.begin() + n_top_candidates,
                         candidates.end(), std::greater<std::pair<float, whisper_token>>());

        for (int k = 0; k < n_top_candidates; k++) {
            decoded.top_candidates.push_back({candidates[k].second, candidates[k].first});
        }

        result.tokens.push_back(decoded);

        // Check for EOT
        if (best_token == token_eot) {
            result.success = true;
            return result;
        }

        // Decode the next token
        if (whisper_decode(ctx, &best_token, 1, n_past, n_threads) != 0) {
            fprintf(stderr, "decode_only: failed to decode token at position %d\n", (int)result.tokens.size());
            return result;
        }

        n_past++;
    }

    result.success = true;
    return result;
}

// Print decode-only results
static void print_decode_result(struct whisper_context * ctx, const DecodeResult & result) {
    printf("\n=== Transcription (decode-only) ===\n");

    // Print text
    std::string text;
    for (const auto & tok : result.tokens) {
        if (tok.id != whisper_token_eot(ctx)) {
            text += whisper_token_to_str(ctx, tok.id);
        }
    }
    printf("%s\n", text.c_str());

    // Print token details
    printf("\n");
    bool first = true;
    for (size_t i = 0; i < result.tokens.size(); i++) {
        const auto & tok = result.tokens[i];
        if (!first) printf(" | ");
        printf("%d,%d,%s,%.4f", (int)i, tok.id,
               whisper_token_to_str(ctx, tok.id), tok.prob);
        first = false;
    }
    printf("\n\n");
}

// Simple line editor with history support
class LineEditor {
private:
    std::vector<std::string> history;
    int history_index;
    struct termios orig_termios;
    bool raw_mode_enabled;

    void enable_raw_mode() {
        tcgetattr(STDIN_FILENO, &orig_termios);
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        raw_mode_enabled = true;
    }

    void disable_raw_mode() {
        if (raw_mode_enabled) {
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
            raw_mode_enabled = false;
        }
    }

public:
    LineEditor() : history_index(-1), raw_mode_enabled(false) {}

    ~LineEditor() {
        disable_raw_mode();
    }

    std::string readline(const char* prompt) {
        printf("%s", prompt);
        fflush(stdout);

        enable_raw_mode();

        std::string line;
        int temp_history_index = history.size();
        std::string current_line;

        while (true) {
            char c;
            if (read(STDIN_FILENO, &c, 1) != 1) break;

            if (c == '\x1b') {  // Escape sequence
                char seq[2];
                if (read(STDIN_FILENO, &seq[0], 1) != 1) break;
                if (read(STDIN_FILENO, &seq[1], 1) != 1) break;

                if (seq[0] == '[') {
                    if (seq[1] == 'A') {  // Arrow up
                        if (temp_history_index > 0) {
                            if (temp_history_index == (int)history.size()) {
                                current_line = line;
                            }
                            temp_history_index--;
                            // Clear current line
                            printf("\r%s", prompt);
                            for (size_t i = 0; i < line.size(); i++) printf(" ");
                            printf("\r%s", prompt);
                            line = history[temp_history_index];
                            printf("%s", line.c_str());
                            fflush(stdout);
                        }
                    } else if (seq[1] == 'B') {  // Arrow down
                        if (temp_history_index < (int)history.size()) {
                            temp_history_index++;
                            // Clear current line
                            printf("\r%s", prompt);
                            for (size_t i = 0; i < line.size(); i++) printf(" ");
                            printf("\r%s", prompt);
                            if (temp_history_index == (int)history.size()) {
                                line = current_line;
                            } else {
                                line = history[temp_history_index];
                            }
                            printf("%s", line.c_str());
                            fflush(stdout);
                        }
                    }
                }
            } else if (c == 127 || c == '\b') {  // Backspace
                if (!line.empty()) {
                    line.pop_back();
                    printf("\b \b");
                    fflush(stdout);
                }
            } else if (c == '\n' || c == '\r') {  // Enter
                printf("\n");
                disable_raw_mode();
                if (!line.empty()) {
                    history.push_back(line);
                }
                return line;
            } else if (c == 4) {  // Ctrl-D (EOF)
                printf("\n");
                disable_raw_mode();
                return "";
            } else if (c >= 32 && c < 127) {  // Printable characters
                line += c;
                printf("%c", c);
                fflush(stdout);
            }
        }

        disable_raw_mode();
        return line;
    }
};

void print_help() {
    printf("\nAvailable commands:\n");
    printf("  help, ?           - Show this help message\n");
    printf("  pos N top [K]     - Show top K candidates at position N (default K=10)\n");
    printf("  pos N id TID      - Force token TID at position N and re-transcribe\n");
    printf("  quit, exit        - Exit the shell\n");
    printf("  Arrow Up/Down     - Navigate command history\n");
    printf("\n");
}

void print_prompt() {
    printf("whsh> ");
    fflush(stdout);
}

// Function to perform transcription and return token map
std::vector<TokenPosition> do_transcription(
    struct whisper_context * ctx,
    const std::vector<float> & pcmf32,
    const std::vector<whisper_token> * prompt_tokens = nullptr
) {
    const int n_threads = 1;

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

    // Capture top candidates for interactive queries
    wparams.capture_top_candidates = true;
    wparams.n_top_candidates       = 20;

    // No VAD
    wparams.vad = false;

    // Set prompt tokens if provided
    if (prompt_tokens != nullptr && !prompt_tokens->empty()) {
        wparams.prompt_tokens = prompt_tokens->data();
        wparams.prompt_n_tokens = prompt_tokens->size();
    }

    // Run the transcription
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stderr, "error: failed to process audio\n");
        return {};
    }

    // Print normal transcription
    printf("\n=== Transcription ===\n");
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        const char * text = whisper_full_get_segment_text(ctx, i);

        printf("[%s --> %s]  %s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);
    }
    printf("\n");

    // Build token position map and print token details in one paragraph
    printf("\n");
    std::vector<TokenPosition> token_map;
    int global_token_pos = 0;
    bool first_token = true;
    for (int i = 0; i < n_segments; i++) {
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; j++) {
            // Store position mapping
            token_map.push_back({i, j});

            const whisper_token token_id = whisper_full_get_token_id(ctx, i, j);
            const char * token_text = whisper_full_get_token_text(ctx, i, j);
            const float token_p = whisper_full_get_token_p(ctx, i, j);

            if (!first_token) {
                printf(" | ");
            }
            printf("%d,%d,%s,%.4f", global_token_pos, token_id, token_text, token_p);
            first_token = false;
            global_token_pos++;
        }
    }
    printf("\n\n");

    return token_map;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    // Check command line arguments
    if (argc != 2) {
        fprintf(stderr, "usage: %s <audio_file>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Interactive whisper shell - transcribes audio then enters interactive mode.\n");
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

    // === ENCODE ONCE ===
    // Convert PCM to mel spectrogram
    fprintf(stderr, "computing mel spectrogram...\n");
    if (whisper_pcm_to_mel(ctx, pcmf32.data(), pcmf32.size(), n_threads) != 0) {
        fprintf(stderr, "error: failed to compute mel spectrogram\n");
        whisper_free(ctx);
        return 5;
    }

    // Encode (populates kv_cross cache - this is the expensive step we do once)
    fprintf(stderr, "encoding audio (populating kv_cross cache)...\n");
    if (whisper_encode(ctx, 0, n_threads) != 0) {
        fprintf(stderr, "error: failed to encode audio\n");
        whisper_free(ctx);
        return 6;
    }
    fprintf(stderr, "encoding complete - kv_cross cache ready for decode-only passes\n");

    // Perform initial transcription using whisper_full
    // Note: whisper_full will re-encode, but this gives us timestamps and proper output
    // For subsequent re-transcriptions, we'll use decode_only which skips encoding
    std::vector<TokenPosition> token_map = do_transcription(ctx, pcmf32);

    if (token_map.empty()) {
        whisper_free(ctx);
        return 7;
    }

    // Track whether we're using whisper_full results or decode_only results
    bool using_decode_only = false;
    DecodeResult current_decode_result;

    // Enter interactive mode
    printf("Entering interactive mode. Type 'help' or '?' for available commands.\n");

    LineEditor editor;

    while (true) {
        // Use line editor for command history support (arrow up/down)
        std::string line = editor.readline("whsh> ");

        // Check for empty (EOF)
        if (line.empty()) {
            break;
        }

        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) {
            // Empty line
            continue;
        }
        line = line.substr(start, end - start + 1);

        // Check for exit commands
        if (line == "quit" || line == "exit") {
            printf("Exiting whsh...\n");
            break;
        }

        // Check for help commands
        if (line == "help" || line == "?") {
            print_help();
            continue;
        }

        // Parse "pos N ..." commands
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "pos") {
            int pos_n;
            std::string subcommand;

            // Read position N
            if (!(iss >> pos_n)) {
                printf("Usage: pos N top [K] or pos N id TID\n");
                continue;
            }

            // Read subcommand (either "top" or "id")
            if (!(iss >> subcommand)) {
                printf("Usage: pos N top [K] or pos N id TID\n");
                continue;
            }

            if (subcommand == "top") {
                // Command: pos N top [K]
                int top_k = 10;  // Default value

                // Try to read K (optional)
                iss >> top_k;  // If this fails, top_k keeps default value of 10

                if (using_decode_only) {
                    // Using decode_only results
                    if (pos_n < 0 || pos_n >= (int)current_decode_result.tokens.size()) {
                        printf("Error: position %d out of range [0, %d]\n", pos_n, (int)current_decode_result.tokens.size() - 1);
                        continue;
                    }

                    const auto & tok = current_decode_result.tokens[pos_n];
                    int k = std::min(top_k, (int)tok.top_candidates.size());

                    printf("Top %d candidates at position %d:\n", k, pos_n);
                    for (int i = 0; i < k; i++) {
                        whisper_token tid = tok.top_candidates[i].first;
                        float prob = tok.top_candidates[i].second;
                        const char * token_text = whisper_token_to_str(ctx, tid);
                        printf("  %d: id=%d token='%s' prob=%.4f\n",
                               i + 1, tid, token_text, prob);
                    }
                } else {
                    // Using whisper_full results
                    if (pos_n < 0 || pos_n >= (int)token_map.size()) {
                        printf("Error: position %d out of range [0, %d]\n", pos_n, (int)token_map.size() - 1);
                        continue;
                    }

                    const TokenPosition& pos = token_map[pos_n];
                    const int n_candidates = whisper_full_n_top_candidates(ctx, pos.segment_idx, pos.token_idx);

                    if (n_candidates == 0) {
                        printf("No candidates available for position %d\n", pos_n);
                        continue;
                    }

                    int k = std::min(top_k, n_candidates);

                    printf("Top %d candidates at position %d:\n", k, pos_n);
                    for (int i = 0; i < k; i++) {
                        whisper_token_candidate cand = whisper_full_get_top_candidate(ctx, pos.segment_idx, pos.token_idx, i);
                        const char * token_text = whisper_token_to_str(ctx, cand.id);
                        printf("  %d: id=%d token='%s' prob=%.4f logprob=%.4f\n",
                               i + 1, cand.id, token_text, cand.p, cand.plog);
                    }
                }
            } else if (subcommand == "id") {
                // Command: pos N id TID
                int token_id;

                if (!(iss >> token_id)) {
                    printf("Usage: pos N id TID\n");
                    continue;
                }

                // Validate position against current token count
                int current_token_count = using_decode_only
                    ? (int)current_decode_result.tokens.size()
                    : (int)token_map.size();

                if (pos_n < 0 || pos_n >= current_token_count) {
                    printf("Error: position %d out of range [0, %d]\n", pos_n, current_token_count - 1);
                    continue;
                }

                printf("Re-transcribing with token %d at position %d (decode-only, reusing kv_cross)...\n", token_id, pos_n);

                // Build forced tokens: all tokens up to pos_n, with token_id at pos_n
                std::vector<whisper_token> forced_tokens;
                for (int i = 0; i <= pos_n; i++) {
                    if (i == pos_n) {
                        // Use the forced token ID
                        forced_tokens.push_back(token_id);
                    } else {
                        // Use the original token ID from current results
                        whisper_token tid;
                        if (using_decode_only) {
                            tid = current_decode_result.tokens[i].id;
                        } else {
                            const TokenPosition& tpos = token_map[i];
                            tid = whisper_full_get_token_id(ctx, tpos.segment_idx, tpos.token_idx);
                        }
                        forced_tokens.push_back(tid);
                    }
                }

                // Use decode_only - this reuses the kv_cross cache from the initial encoding
                // No re-encoding needed - kv_cross persists in the state
                current_decode_result = decode_only(ctx, forced_tokens);

                if (!current_decode_result.success) {
                    printf("Re-transcription failed\n");
                    continue;
                }

                // Print the decode-only results
                print_decode_result(ctx, current_decode_result);

                // Switch to decode_only mode for subsequent queries
                using_decode_only = true;
            } else {
                printf("Unknown subcommand: '%s'. Usage: pos N top [K] or pos N id TID\n", subcommand.c_str());
            }
        } else if (!line.empty()) {
            printf("Unknown command: '%s'. Type 'help' or '?' for available commands.\n", line.c_str());
        }
    }

    // Clean up
    whisper_free(ctx);

    return 0;
}
