# Perplexity SSE `step_type` Enum — Extracted from SPA Bundle

**Source**: Perplexity web SPA bundle, captured January 2026  
**Primary file**: `ThreadEntryContext-hgdcVwpW.js` (19KB minified) — contains the complete `??` content-field fallback chain  
**Secondary**: `mission-control-page-CMVaqG1M.js` (step_type dispatch), `pplx-stream-BSN55UYQ.js` (INITIAL_QUERY construction), `StepRenderer-DrvDub-b.js` (334KB step renderer)

---

## The Canonical Content-Field Fallback Chain

This is the complete `??` chain from `ThreadEntryContext-hgdcVwpW.js` that maps each step's `step_type`
to its typed content field. Every `*_content` field name corresponds 1:1 with a `step_type` value
(by convention, `UPPER_CASE` step_type → `lower_case_content` field).

```javascript
// Verbatim from ThreadEntryContext-hgdcVwpW.js:
{
  step_type: r.step_type,
  uuid: r.uuid ?? "",
  content: r?.initial_query_content                     // 1
        ?? r?.attachment_content                         // 2
        ?? r?.terminate_content                          // 3
        ?? r?.search_web_content                         // 4
        ?? r?.web_results_content                        // 5
        ?? r?.code_content                               // 6
        ?? r?.table_status_content                       // 7
        ?? r?.entropy_request_content                    // 8
        ?? r?.thought_content                            // 9
        ?? r?.browser_search_content                     // 10
        ?? r?.browser_open_tab_content                   // 11
        ?? r?.browser_open_tab_results_content           // 12
        ?? r?.url_navigate_content                       // 13
        ?? r?.browser_get_site_content_content           // 14
        ?? r?.user_clarification_content                 // 15
        ?? r?.browser_get_history_summary_content        // 16
        ?? r?.browser_get_open_tab_content_content       // 17
        ?? r?.read_calendar_content                      // 18
        ?? r?.read_calendar_response_content             // 19
        ?? r?.read_email_content                         // 20
        ?? r?.read_email_response_content                // 21
        ?? r?.update_calendar_content                    // 22
        ?? r?.generate_image_content                     // 23
        ?? r?.generate_image_results_content             // 24
        ?? r?.generate_video_content                     // 25
        ?? r?.generate_video_results_content             // 26
        ?? r?.search_tabs_content                        // 27
        ?? r?.search_tabs_results_content                // 28
        ?? r?.create_app_results_content                 // 29
        ?? r?.browser_close_tabs_content                 // 30
        ?? r?.browser_close_tabs_results_content         // 31
        ?? r?.update_calendar_response_content           // 32
        ?? r?.browser_group_tabs_content                 // 33
        ?? r?.browser_group_tabs_results_content         // 34
        ?? r?.create_chart_content                       // 35
        ?? r?.get_url_content_content                    // 36
        ?? r?.create_client_app_content                  // 37
        ?? r?.get_user_info_content                      // 38
        ?? r?.get_user_info_response_content             // 39
        ?? r?.get_free_busy_content                      // 40
        ?? r?.get_free_busy_response_content             // 41
        ?? r?.send_email_content                         // 42
        ?? r?.send_email_response_content                // 43
        ?? r?.browser_ungroup_content                    // 44
        ?? r?.browser_search_tab_groups_content          // 45
        ?? r?.browser_search_tab_groups_result_content   // 46
        ?? r?.search_browser_content                     // 47
        ?? r?.search_browser_results_content             // 48
        ?? r?.clarifying_questions_content               // 49
        ?? r?.clarifying_questions_output_content        // 50
        ?? r?.email_calendar_agent_content               // 51
        ?? r?.email_calendar_agent_response_content      // 52
        ?? r?.mcp_tool_input_content                     // 53
        ?? r?.mcp_tool_output_content                    // 54
        ?? r?.research_clarifying_questions_content      // 55
        ?? r?.create_tasks_content                       // 56
        ?? r?.create_tasks_response_content              // 57
        ?? r?.flights_search_content                     // 58
        ?? r?.flights_booking_content                    // 59
        ?? r?.flights_search_response_content            // 60
        ?? r?.flights_booking_response_content           // 61
        ?? r?.flights_agent_content                      // 62
        ?? r?.canvas_agent_content                       // 63
        ?? r?.comet_agent_tool_input_content             // 64
        ?? r?.comet_agent_tool_output_content            // 65
        ?? r?.connector_direct_search_con[...]           // 66 (truncated)
}
```

---

## Complete step_type Enum — All 65+ Values by Category

### Core Query Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 1 | `INITIAL_QUERY` | `initial_query_content` | Echoes user prompt; "Starting up" animation in UI | SPA + wire |
| 2 | `FINAL` | `final_content` | Final assembled answer (also in `markdown_block`) | SPA + wire + OSS |
| 3 | `TERMINATE` | `terminate_content` | Goal termination / early stop signal | SPA + types |
| 4 | `ATTACHMENT` | `attachment_content` | File attachment processing | SPA only |

### Web Search Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 5 | `SEARCH_WEB` | `search_web_content` | Web search query dispatched | SPA + wire + 5 OSS repos |
| 6 | `WEB_RESULTS` | `web_results_content` | Web search results received | SPA + types |
| 7 | `SEARCH_RESULTS` | (unknown) | Search results aggregation (separate from WEB_RESULTS) | SPA only |

### Deep Research / Mission Control Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 8 | `ENTROPY_REQUEST` | `entropy_request_content` | Agent task dispatch with `tasks[].agent_messages[]` | SPA only |
| 9 | `THOUGHT` | `thought_content` | Agent reasoning/thought step | SPA only |
| 10 | `USER_CLARIFICATION` | `user_clarification_content` | Response to agent clarification request | SPA only |
| 11 | `RESEARCH_CLARIFYING_QUESTIONS` | `research_clarifying_questions_content` | Deep Research clarification request | SPA + wire + OSS |
| 12 | `CLARIFYING_QUESTIONS` | `clarifying_questions_content` | General clarifying question from model | SPA only |
| 13 | `CLARIFYING_QUESTIONS_OUTPUT` | `clarifying_questions_output_content` | User's clarification answer | SPA only |
| 14 | `COMET_AGENT_TOOL_INPUT` | `comet_agent_tool_input_content` | Comet agent invocation; `task_uuid` | SPA only |
| 15 | `COMET_AGENT_TOOL_OUTPUT` | `comet_agent_tool_output_content` | Comet agent result | SPA only |

### Browser Agent Steps (Deep Research browser mode)

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 16 | `BROWSER_SEARCH` | `browser_search_content` | Browser agent search query | SPA only |
| 17 | `BROWSER_OPEN_TAB` | `browser_open_tab_content` | Open new browser tab | SPA only |
| 18 | `BROWSER_OPEN_TAB_RESULTS` | `browser_open_tab_results_content` | Tab opened with URL | SPA only |
| 19 | `URL_NAVIGATE` | `url_navigate_content` | Navigate to URL | SPA only |
| 20 | `BROWSER_GET_SITE_CONTENT` | `browser_get_site_content_content` | Extract page content | SPA only |
| 21 | `BROWSER_GET_HISTORY_SUMMARY` | `browser_get_history_summary_content` | Browser history summary | SPA only |
| 22 | `BROWSER_GET_OPEN_TAB_CONTENT` | `browser_get_open_tab_content_content` | Get open tab content | SPA only |
| 23 | `BROWSER_CLOSE_TABS` | `browser_close_tabs_content` | Close browser tabs | SPA only |
| 24 | `BROWSER_CLOSE_TABS_RESULTS` | `browser_close_tabs_results_content` | Tab close results | SPA only |
| 25 | `BROWSER_GROUP_TABS` | `browser_group_tabs_content` | Group browser tabs | SPA only |
| 26 | `BROWSER_GROUP_TABS_RESULTS` | `browser_group_tabs_results_content` | Tab grouping results | SPA only |
| 27 | `BROWSER_UNGROUP` | `browser_ungroup_content` | Ungroup browser tabs | SPA only |
| 28 | `BROWSER_SEARCH_TAB_GROUPS` | `browser_search_tab_groups_content` | Search tab groups | SPA only |
| 29 | `BROWSER_SEARCH_TAB_GROUPS_RESULT` | `browser_search_tab_groups_result_content` | Tab group search results | SPA only |
| 30 | `SEARCH_BROWSER` | `search_browser_content` | Alternative browser search | SPA only |
| 31 | `SEARCH_BROWSER_RESULTS` | `search_browser_results_content` | Browser search results | SPA only |
| 32 | `SEARCH_TABS` | `search_tabs_content` | Search across tabs | SPA only |
| 33 | `SEARCH_TABS_RESULTS` | `search_tabs_results_content` | Tab search results | SPA only |
| 34 | `GET_URL_CONTENT` | `get_url_content_content` | Get content from URL | SPA only |

### **MCP Tool Call Steps (Connectors)**

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 35 | **`MCP_TOOL_INPUT`** | `mcp_tool_input_content` | **MCP tool invocation (request)** | **SPA + wire** |
| 36 | **`MCP_TOOL_OUTPUT`** | `mcp_tool_output_content` | **MCP tool execution result** | **SPA + wire** |

### Calendar / Email Agent Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 37 | `READ_CALENDAR` | `read_calendar_content` | Read calendar events | SPA only |
| 38 | `READ_CALENDAR_RESPONSE` | `read_calendar_response_content` | Calendar read results | SPA only |
| 39 | `UPDATE_CALENDAR` | `update_calendar_content` | Create/update calendar event | SPA only |
| 40 | `UPDATE_CALENDAR_RESPONSE` | `update_calendar_response_content` | Calendar update result | SPA only |
| 41 | `READ_EMAIL` | `read_email_content` | Read email messages | SPA only |
| 42 | `READ_EMAIL_RESPONSE` | `read_email_response_content` | Email read results | SPA only |
| 43 | `SEND_EMAIL` | `send_email_content` | Send email | SPA only |
| 44 | `SEND_EMAIL_RESPONSE` | `send_email_response_content` | Email send result | SPA only |
| 45 | `GET_USER_INFO` | `get_user_info_content` | Get user profile info | SPA only |
| 46 | `GET_USER_INFO_RESPONSE` | `get_user_info_response_content` | User info response | SPA only |
| 47 | `GET_FREE_BUSY` | `get_free_busy_content` | Check calendar availability | SPA only |
| 48 | `GET_FREE_BUSY_RESPONSE` | `get_free_busy_response_content` | Free/busy results | SPA only |
| 49 | `EMAIL_CALENDAR_AGENT` | `email_calendar_agent_content` | Combined email+calendar agent | SPA only |
| 50 | `EMAIL_CALENDAR_AGENT_RESPONSE` | `email_calendar_agent_response_content` | Agent response | SPA only |

### Image / Video Generation Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 51 | `GENERATE_IMAGE` | `generate_image_content` | Image generation prompt | SPA only |
| 52 | `GENERATE_IMAGE_RESULTS` | `generate_image_results_content` | Generated image URLs | SPA + OSS (polychat) |
| 53 | `GENERATE_VIDEO` | `generate_video_content` | Video generation prompt | SPA only |
| 54 | `GENERATE_VIDEO_RESULTS` | `generate_video_results_content` | Generated video URLs | SPA only |

### Flights / Travel Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 55 | `FLIGHTS_SEARCH` | `flights_search_content` | Flight search query | SPA only |
| 56 | `FLIGHTS_BOOKING` | `flights_booking_content` | Flight booking action | SPA only |
| 57 | `FLIGHTS_SEARCH_RESPONSE` | `flights_search_response_content` | Flight search results | SPA only |
| 58 | `FLIGHTS_BOOKING_RESPONSE` | `flights_booking_response_content` | Booking confirmation | SPA only |
| 59 | `FLIGHTS_AGENT` | `flights_agent_content` | Combined flights agent | SPA only |

### Productivity Steps

| # | step_type | content field | Description | Verified |
|---|-----------|---------------|-------------|----------|
| 60 | `CREATE_TASKS` | `create_tasks_content` | Create task action | SPA only |
| 61 | `CREATE_TASKS_RESPONSE` | `create_tasks_response_content` | Task creation result | SPA only |
| 62 | `TABLE_STATUS` | `table_status_content` | Table rendering status | SPA only |
| 63 | `CODE` | `code_content` | Code execution step | SPA only |
| 64 | `CREATE_CHART` | `create_chart_content` | Chart generation | SPA only |
| 65 | `CANVAS_AGENT` | `canvas_agent_content` | Canvas/drawing agent | SPA only |
| 66 | `CREATE_APP_RESULTS` | `create_app_results_content` | App creation results | SPA only |
| 67 | `CREATE_CLIENT_APP` | `create_client_app_content` | Client app creation | SPA only |
| 68 | `CONNECTOR_DIRECT_SEARCH` | `connector_direct_search_con[...]` | Direct connector file search | SPA only |

---

## MCP_TOOL_INPUT Content Shape

From `ThreadEntryContext-hgdcVwpW.js` field chain + wire captures:

```typescript
{
  step_type: "MCP_TOOL_INPUT",
  uuid: string,
  mcp_tool_input_content: {
    goal_id: string,                    // pairs with MCP_TOOL_OUTPUT
    tool_id: string,                    // e.g. "get_me", "list_pull_requests"
    tool_name: string,                  // e.g. "get_me"
    tool_args: Record<string, any>,     // tool input arguments
    authenticated: boolean,
    app: string,                        // e.g. "GitHub", "Slack", "Notion"
    mcp_server_type: string,            // "MCP_SERVER_TYPE_REMOTE" (only observed value)
    source_type: string,                // e.g. "github_mcp_direct"
    tool_input_summary: string,         // Human-readable summary for UI card
    request_user_approval: {
      uuid: string,
      request_user_approval: boolean    // true → stream pauses for user approval
    },
    approval_result: null | {
      // Set when user approves/rejects via /rest/sse/handle_tool_user_approval_response
      // Exact shape unknown — not in this SPA capture
    },
    logo_url: string                    // CDN URL for connector icon branding
  }
}
```

## MCP_TOOL_OUTPUT Content Shape

```typescript
{
  step_type: "MCP_TOOL_OUTPUT",
  uuid: string,
  mcp_tool_output_content: {
    goal_id: string,              // pairs with MCP_TOOL_INPUT
    status: "success" | string,   // success | error variants (specifics unknown)
    content: string,              // JSON-encoded tool result string
    should_rerun_query: boolean,  // tool result may trigger re-query
    app: string,
    authenticated: boolean,
    logo_url: string,
    data_is_redacted: null | boolean
  }
}
```

---

## Additional SPA Modules Identified

From `perplexity_spa_full_spec.json` asset index:

| Module | Size | Relevance |
|--------|------|-----------|
| `ThreadEntryContext-hgdcVwpW.js` | 19KB | **Canonical source** — complete content-field chain |
| `StepRenderer-DrvDub-b.js` | 334KB | Step rendering UI (chart/graph components dominate) |
| `pplx-stream-BSN55UYQ.js` | ~10KB | SSE stream construction, INITIAL_QUERY injection |
| `mission-control-page-CMVaqG1M.js` | ~15KB | Mission Control UI with ENTROPY_REQUEST dispatch |
| `MultiStepProvider-BIEI167b.js` | 1KB | Multi-step search provider (thin wrapper) |
| `connectors-Bc53l23-.js` | — | Connector listing with `github_mcp_direct` references |
| `connectors-BO3LWElm.js` | — | Connector infrastructure |
| `connectorDetails-BjBm-BEZ.js` | — | Individual connector detail view |

---

## OSS Coverage Gap Summary

| Category | Count | OSS Handled | MCP-Aware OSS |
|----------|-------|-------------|---------------|
| Core query steps | 4 | 2 (INITIAL_QUERY typed, FINAL handled) | 0 |
| Web search steps | 3 | 2 (SEARCH_WEB, WEB_RESULTS) | 0 |
| Deep Research steps | 8 | 1 (RESEARCH_CLARIFYING_QUESTIONS) | 0 |
| Browser agent steps | 19 | 0 | 0 |
| **MCP tool steps** | **2** | **0** | **0** |
| Calendar/email steps | 14 | 0 | 0 |
| Image/video steps | 4 | 1 (GENERATE_IMAGE_RESULTS in polychat) | 0 |
| Flights steps | 5 | 0 | 0 |
| Productivity steps | 9 | 0 | 0 |
| **TOTAL** | **68** | **6 (9%)** | **0** |

**Bottom line**: Open-source covers 9% of Perplexity's step_type surface. MCP tool handling is at absolute zero. This SPA bundle extraction provides the complete canonical enum — ready for ccproxy implementation.
