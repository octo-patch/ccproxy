{
  settings = {
    host = "127.0.0.1";
    port = 4000;
    log_level = "INFO";
    providers = {
      anthropic = {
        auth = {
          type = "command";
          command = "jq -r '.claudeAiOauth.accessToken' ~/.claude/.credentials.json";
        };
        host = "api.anthropic.com";
        path = "/v1/messages";
        type = "anthropic";
      };
      gemini = {
        auth = {
          type = "google_oauth";
          client_id = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
          client_secret = "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";
        };
        host = "cloudcode-pa.googleapis.com";
        path = "/v1internal:{action}";
        type = "gemini";
      };
      deepseek = {
        auth = {
          type = "command";
          command = "printenv DEEPSEEK_API_KEY";
          header = "x-api-key";
        };
        host = "api.deepseek.com";
        path = "/anthropic/v1/messages";
        type = "anthropic";
      };
      perplexity_pro = {
        auth = {
          type = "file";
          file = "~/.opnix/secrets/perplexity-pro-api-key";
        };
        host = "www.perplexity.ai";
        path = "/rest/sse/perplexity_ask";
        type = "perplexity_pro";
        fingerprint_profile = "chrome131";
      };
      codex = {
        # Routes Codex CLI traffic to OpenAI's ChatGPT-backed Responses
        # endpoint. ``auth_mode=chatgpt`` in ~/.codex/auth.json means
        # Codex hits chatgpt.com/backend-api/codex (not api.openai.com),
        # bearing the JWT ``access_token`` from that file.
        # Inbound /v1/responses matches provider type ``openai_responses``
        # so the transform router auto-derives a same-format redirect —
        # no cross-format transform fires.
        auth = {
          type = "command";
          command = "jq -r '.tokens.access_token' ~/.codex/auth.json";
        };
        host = "chatgpt.com";
        path = "/backend-api/codex/responses";
        type = "openai_responses";
      };
    };
    hooks = {
      inbound = [
        "ccproxy.hooks.forward_oauth"
        "ccproxy.hooks.extract_session_id"
        "ccproxy.hooks.extract_pplx_files"
        "ccproxy.hooks.pplx_thread_inject"
      ];
      outbound = [
        "ccproxy.hooks.gemini_cli"
        "ccproxy.hooks.pplx_stamp_headers"
        "ccproxy.hooks.pplx_preflight"
        "ccproxy.hooks.inject_mcp_notifications"
        "ccproxy.hooks.verbose_mode"
        "ccproxy.hooks.commitbee_compat"
        "ccproxy.hooks.shape"
      ];
    };
    pplx = {
      search = {
        language = "en-US";
        timezone = "America/Los_Angeles";
        search_focus = "internet";
        sources = [ "web" ];
        search_recency_filter = null;
        is_incognito = false;
        skip_search_enabled = true;
        is_nav_suggestions_disabled = true;
        always_search_override = false;
        override_no_search = false;
        preflight_timeout_seconds = 5;
      };
      thread = {
        consistency_mode = "warn";
        citation_mode = "markdown";
        ttl_seconds = 1800;
        fetch_page_size = 100;
        fetch_timeout_seconds = 10;
      };
      upload = {
        max_files = 30;
        max_file_size_bytes = 52428800;
        fetch_timeout_seconds = 10;
        upload_timeout_seconds = 60;
        subscribe_timeout_seconds = 120;
      };
    };
    gemini_capacity = {
      enabled = true;
      retry_status_codes = [ 429 503 500 ];
      fallback_models = [ "gemini-3-flash-preview" "gemini-2.5-pro" "gemini-2.5-flash" ];
      sticky_retry_attempts = 3;
      sticky_retry_max_delay_seconds = 60;
      terminal_delay_threshold_seconds = 300;
      total_retry_budget_seconds = 120;
    };
    otel = {
      enabled = false;
      endpoint = "http://localhost:4317";
      service_name = "ccproxy";
    };
    mcp = {
      http = {
        enabled = true;
        host = "127.0.0.1";
        port = 4030;
        auth = null;
      };
      buffer = {
        max_events_per_task = 65536;
        ttl_seconds = 600;
      };
    };
    oauth = {
      command_timeout_seconds = 5;
      refresh_timeout_seconds = 15;
      refresh_headroom_seconds = 60;
    };
    shaping = {
      enabled = true;
      shapes_dir = "~/.config/ccproxy/shaping/shapes";
      patches_dir = "~/.config/ccproxy/shaping/patches";
      providers = {
        anthropic = {
          content_fields = [
            "model" "messages" "tools" "tool_choice" "system" "thinking" "context_management"
            "stream" "max_tokens" "temperature" "top_p" "top_k" "stop_sequences"
          ];
          merge_strategies = { system = "prepend_shape:2"; };
          shape_hooks = [
            "ccproxy.shaping.regenerate"
            {
              hook = "ccproxy.shaping.caching.strip";
              params = {
                paths = [ "system.*.cache_control" ];
              };
            }
            {
              hook = "ccproxy.shaping.caching.insert";
              params = {
                path = "system.-1.cache_control";
                value = {
                  type = "ephemeral";
                };
              };
            }
          ];
          preserve_headers = [ "authorization" "x-api-key" "x-goog-api-key" "host" ];
          strip_headers = [
            "authorization" "x-api-key" "x-goog-api-key"
            "content-length" "host" "transfer-encoding" "connection"
            "accept-encoding"
          ];
          capture = { path_pattern = "^/v1/messages"; };
        };
        gemini = {
          content_fields = [ "model" "project" ];
          shape_hooks = [
            "ccproxy.shaping.regenerate"
            "ccproxy.shaping.gemini"
          ];
          preserve_headers = [ "authorization" "host" ];
          strip_headers = [
            "authorization" "content-length" "host"
            "transfer-encoding" "connection" "accept-encoding"
          ];
          capture = { path_pattern = "^/v1internal:"; };
        };
      };
    };
    inspector = {
      port = 8083;
      cert_dir = "~/.config/ccproxy";
      transforms = [];
    };
  };
}
