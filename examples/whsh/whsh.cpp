#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <sstream>
#include <termios.h>
#include <unistd.h>

// Interactive whisper shell (whsh)
// Usage: whsh [-m model_path] [-l lang] <audio_file>
// After transcription, enters interactive mode with commands:
// - help, ? : Show available commands
// - pos N top [K] : Show top K candidate tokens at position N (default K=10)
// - pos N id TID : Force token TID at position N and re-transcribe from that point
// - pos N new : Take first N tokens and re-transcribe in new context
// - tok <text> : Tokenize text with and without leading space
// - lang [code] : Show/change language and re-transcribe (e.g., 'en', 'de', 'de-en' for translation)
// - quit, exit : Exit the shell
// - Arrow Up/Down : Navigate command history
//
// Architecture: Encode-once, decode-many
// - Audio is encoded once at startup (populates kv_cross cache)
// - Re-transcriptions use whisper_full with skip_encode=true (reuses kv_cross)

// Structure to map global token position to segment/token indices
struct TokenPosition {
    int segment_idx;
    int token_idx;
};

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
    printf("  pos N new         - Take first N tokens and re-transcribe in new context\n");
    printf("  tok <text>        - Tokenize text with and without leading space\n");
    printf("  lang [code]       - Show/change language and re-transcribe\n");
    printf("                      Use language code (e.g., 'en', 'de', 'fr') for transcription\n");
    printf("                      Use 'code-en' (e.g., 'de-en') for translation to English\n");
    printf("  quit, exit        - Exit the shell\n");
    printf("  Arrow Up/Down     - Navigate command history\n");
    printf("\n");
}

void print_prompt() {
    printf("whsh> ");
    fflush(stdout);
}

// Function to perform transcription and return token map
// forced_tokens: tokens to force at the start of decoding (for re-transcription with alternatives)
// skip_encode: if true, reuse kv_cross from previous encode (for re-transcription)
// lang: language code (e.g., "en", "de", "fr") or "code-en" for automatic translation
std::vector<TokenPosition> do_transcription(
    struct whisper_context * ctx,
    const std::vector<float> & pcmf32,
    const char * lang,
    const std::vector<whisper_token> * forced_tokens = nullptr,
    bool skip_encode = false
) {
    const int n_threads = 1;

    // Set up whisper parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;

    // Parse language parameter: if it ends with "-en", enable translation
    std::string lang_str(lang);
    if (lang_str.size() >= 3 && lang_str.substr(lang_str.size() - 3) == "-en") {
        // Extract the source language code (before "-en")
        std::string source_lang = lang_str.substr(0, lang_str.size() - 3);
        wparams.translate = true;
        wparams.language = source_lang.c_str();
    } else {
        wparams.translate = false;
        wparams.language = lang;
    }
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

    // Skip encoding if requested (reuse kv_cross from previous encode)
    wparams.skip_encode = skip_encode;

    // Set forced tokens if provided (for exploring alternative transcriptions)
    if (forced_tokens != nullptr && !forced_tokens->empty()) {
        wparams.forced_tokens = forced_tokens->data();
        wparams.forced_n_tokens = forced_tokens->size();
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

    // Print version on startup
    fprintf(stderr, "whsh version %s\n", whisper_version());

    // Default model path and language
    const char * model_path = "models/ggml-tiny.en.bin";
    const char * fname_inp = nullptr;
    std::string language = "en";  // Default language (mutable for in-app lang command)

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                fprintf(stderr, "error: -m requires a model path argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lang") == 0) {
            if (i + 1 < argc) {
                language = std::string(argv[++i]);
            } else {
                fprintf(stderr, "error: -l/--lang requires a language code argument\n");
                return 1;
            }
        } else {
            fname_inp = argv[i];
        }
    }

    // Check if audio file was provided
    if (fname_inp == nullptr) {
        fprintf(stderr, "usage: %s [-m model_path] [-l lang] <audio_file>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Interactive whisper shell - transcribes audio then enters interactive mode.\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -m <model_path>  Path to model file (default: models/ggml-tiny.en.bin)\n");
        fprintf(stderr, "  -l, --lang <code> Language code for transcription (default: en)\n");
        fprintf(stderr, "                   Use language code (e.g., 'en', 'de', 'fr') for transcription\n");
        fprintf(stderr, "                   Use 'code-en' for automatic translation to English\n");
        fprintf(stderr, "Fixed settings: CPU only, single thread, no VAD\n");
        fprintf(stderr, "Supported audio formats: flac, mp3, ogg, wav\n");
        return 1;
    }

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

    // Perform initial transcription (this encodes + decodes)
    // The encoding result (kv_cross) will be reused for re-transcriptions via skip_encode
    std::vector<TokenPosition> token_map = do_transcription(ctx, pcmf32, language.c_str());

    if (token_map.empty()) {
        whisper_free(ctx);
        return 5;
    }

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

            // Read subcommand (either "top", "id", or "new")
            if (!(iss >> subcommand)) {
                printf("Usage: pos N top [K], pos N id TID, or pos N new\n");
                continue;
            }

            if (subcommand == "top") {
                // Command: pos N top [K]
                int top_k = 10;  // Default value

                // Try to read K (optional)
                iss >> top_k;  // If this fails, top_k keeps default value of 10

                // Validate position
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
            } else if (subcommand == "id") {
                // Command: pos N id TID
                int token_id;

                if (!(iss >> token_id)) {
                    printf("Usage: pos N id TID\n");
                    continue;
                }

                // Validate position
                if (pos_n < 0 || pos_n >= (int)token_map.size()) {
                    printf("Error: position %d out of range [0, %d]\n", pos_n, (int)token_map.size() - 1);
                    continue;
                }

                printf("Re-transcribing with token %d forced at position %d (skip_encode + forced_tokens)...\n", token_id, pos_n);

                // Build forced tokens: all tokens up to pos_n, with token_id at pos_n
                std::vector<whisper_token> forced_tokens;
                for (int i = 0; i <= pos_n; i++) {
                    const TokenPosition& tpos = token_map[i];
                    if (i == pos_n) {
                        forced_tokens.push_back(token_id);
                    } else {
                        forced_tokens.push_back(whisper_full_get_token_id(ctx, tpos.segment_idx, tpos.token_idx));
                    }
                }

                // Re-transcribe with skip_encode=true and forced_tokens
                token_map = do_transcription(ctx, pcmf32, language.c_str(), &forced_tokens, true);

                if (token_map.empty()) {
                    printf("Re-transcription failed\n");
                }
            } else if (subcommand == "new") {
                // Command: pos N new
                // Take first N tokens and re-transcribe in new context

                // Validate position
                if (pos_n < 0 || pos_n >= (int)token_map.size()) {
                    printf("Error: position %d out of range [0, %d]\n", pos_n, (int)token_map.size() - 1);
                    continue;
                }

                printf("Re-transcribing with tokens 0-%d in new context (fresh encoding + forced_tokens)...\n", pos_n);

                // Build forced tokens: take tokens 0 to N (inclusive)
                std::vector<whisper_token> forced_tokens;
                for (int i = 0; i <= pos_n; i++) {
                    const TokenPosition& tpos = token_map[i];
                    forced_tokens.push_back(whisper_full_get_token_id(ctx, tpos.segment_idx, tpos.token_idx));
                }

                // Re-transcribe with skip_encode=false (fresh encoding) and forced_tokens
                token_map = do_transcription(ctx, pcmf32, language.c_str(), &forced_tokens, false);

                if (token_map.empty()) {
                    printf("Re-transcription failed\n");
                }
            } else {
                printf("Unknown subcommand: '%s'. Usage: pos N top [K], pos N id TID, or pos N new\n", subcommand.c_str());
            }
        } else if (cmd == "tok") {
            // Command: tok <text>
            // Tokenize the remaining text with and without leading space

            // Get the rest of the line as the text to tokenize
            std::string text;
            std::getline(iss, text);

            // Trim leading whitespace from text
            size_t start = text.find_first_not_of(" \t");
            if (start == std::string::npos) {
                printf("Usage: tok <text>\n");
                continue;
            }
            text = text.substr(start);

            if (text.empty()) {
                printf("Usage: tok <text>\n");
                continue;
            }

            // Tokenize without leading space
            std::vector<whisper_token> tokens1(text.size() + 10);
            int n_tokens1 = whisper_tokenize(ctx, text.c_str(), tokens1.data(), tokens1.size());
            tokens1.resize(n_tokens1);

            // Tokenize with leading space
            std::string text_with_space = " " + text;
            std::vector<whisper_token> tokens2(text_with_space.size() + 10);
            int n_tokens2 = whisper_tokenize(ctx, text_with_space.c_str(), tokens2.data(), tokens2.size());
            tokens2.resize(n_tokens2);

            // Print results
            printf("\nTokenization of '%s':\n", text.c_str());
            printf("  Without space (%d tokens):", n_tokens1);
            for (int i = 0; i < n_tokens1; i++) {
                printf(" %d='%s'", tokens1[i], whisper_token_to_str(ctx, tokens1[i]));
            }
            printf("\n");

            printf("Tokenization of ' %s':\n", text.c_str());
            printf("  With space    (%d tokens):", n_tokens2);
            for (int i = 0; i < n_tokens2; i++) {
                printf(" %d='%s'", tokens2[i], whisper_token_to_str(ctx, tokens2[i]));
            }
            printf("\n\n");
        } else if (cmd == "lang") {
            // Command: lang <code>
            // Change language and re-transcribe

            std::string new_lang;
            if (!(iss >> new_lang)) {
                printf("Current language: %s\n", language.c_str());
                printf("Usage: lang <code>\n");
                printf("  <code> can be a language code (e.g., 'en', 'de', 'fr') for transcription\n");
                printf("  or 'code-en' (e.g., 'de-en', 'fr-en') for automatic translation to English\n");
                continue;
            }

            printf("Changing language from '%s' to '%s' and re-transcribing...\n", language.c_str(), new_lang.c_str());
            language = new_lang;

            // Re-transcribe with new language (skip_encode=false to get fresh encoding)
            token_map = do_transcription(ctx, pcmf32, language.c_str(), nullptr, false);

            if (token_map.empty()) {
                printf("Re-transcription failed\n");
            }
        } else if (!line.empty()) {
            printf("Unknown command: '%s'. Type 'help' or '?' for available commands.\n", line.c_str());
        }
    }

    // Clean up
    whisper_free(ctx);

    return 0;
}
