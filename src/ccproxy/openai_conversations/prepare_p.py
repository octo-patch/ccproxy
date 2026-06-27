"""25-slot config builder for the OpenAI Conversations sentinel p field.

Ported from the MIT-licensed gproxy chatgpt channel:
  Copyright (c) 2026 LeenHawk
  https://github.com/LeenHawk/gproxy  (MIT License)

Cross-checked against aurora-develop/aurora ``internal/fingerprint/build25.go``
for the current 2026-06 25-element layout and ``internal/prooftoken/prooftoken.go``
for requirements-token vs proof-token slot assignment rules.

Slot layout (Build25, 2026-06 specimens):
  [0]  str: screen.width + screen.height (as string)
  [1]  str: new Date().toString()
  [2]  str: jsHeapSizeLimit (as string)
  [3]  num: Math.random() for prepare; nonce (int) for PoW
  [4]  num: Math.random()
  [5]  str: navigator.userAgent
  [6]  str: currentScript.src
  [7]  str: documentElement data-build id
  [8]  str: navigator.language
  [9]  num: Math.random() for prepare; elapsed ms (int) for PoW
  [10] str: "X in navigator" probe ("name-[object Name]", U+2212 separator)
  [11] str: Object.keys(document) random key
  [12] str: Object.getOwnPropertyNames(window) random key
  [13] num: performance.now()
  [14] str: device_id
  [15] str: URLSearchParams(location.search) joined keys
  [16] num: navigator.hardwareConcurrency
  [17] num: performance.timeOrigin (unix ms)
  [18-24] num: "X in window" probes (7 values, default 0)
"""

from __future__ import annotations

import base64
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Canonical defaults from 2026-04/2026-06 chatgpt.com captures.
_DEFAULT_BUILD_ID = "prod-d7545204e22cb990d0245281e6550977d93b6a81"
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
)
_DEFAULT_SCRIPT_SRC = "https://chatgpt.com/_next/static/chunks/prod-d7545204e22cb990d0245281e6550977d93b6a81.js"

# produced by the browser's "X in navigator" probe (chatgpt.com SDK).
# They must not be changed to ASCII hyphens.
_DEFAULT_NAVIGATOR_PROBES = (
    "windowControlsOverlay−[object WindowControlsOverlay]",  # noqa: RUF001
    "geolocation−[object Geolocation]",  # noqa: RUF001
    "clipboard−[object Clipboard]",  # noqa: RUF001
    "mediaDevices−[object MediaDevices]",  # noqa: RUF001
    "permissions−[object Permissions]",  # noqa: RUF001
    "bluetooth−[object Bluetooth]",  # noqa: RUF001
    "usb−[object USB]",  # noqa: RUF001
    "serial−[object Serial]",  # noqa: RUF001
    "hid−[object HID]",  # noqa: RUF001
    "presentation−[object Presentation]",  # noqa: RUF001
    "credentials−[object CredentialsContainer]",  # noqa: RUF001
)
_DEFAULT_DOCUMENT_KEYS = ("location", "_reactListening7emk2nodhb")
_DEFAULT_WINDOW_KEYS = (
    "outerWidth",
    "__oai_so_kp",
    "localStorage",
    "visualViewport",
)


def _format_browser_date(unix_secs: float, tz_offset_minutes: int, tz_label: str) -> str:
    """Format a unix timestamp as a browser Date.toString() string.

    Matches the format produced by Chrome/Edge on Windows:
    "Tue Apr 21 2026 17:25:57 GMT+0800 (中国标准时间)"
    """
    shifted = int(unix_secs) + tz_offset_minutes * 60
    weekdays = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

    # Civil calendar from days (Howard Hinnant algorithm)
    days = shifted // 86400
    rem = shifted % 86400
    hh = rem // 3600
    mm = (rem % 3600) // 60
    ss = rem % 60
    wd = (days + 4) % 7  # 1970-01-01 was Thursday

    z = days + 719_468
    era = z // 146_097
    doe = z - era * 146_097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    if m <= 2:
        y += 1

    sign = "+" if tz_offset_minutes >= 0 else "-"
    abs_off = abs(tz_offset_minutes)
    return (
        f"{weekdays[wd % 7]} {months[m - 1]} {d:02d} {y} "
        f"{hh:02d}:{mm:02d}:{ss:02d} GMT{sign}{abs_off // 60:02d}{abs_off % 60:02d} "
        f"({tz_label})"
    )


@dataclass
class ConfigOptions:
    """Options for building the 25-slot fingerprint config array.

    Use :meth:`browser_default` for runtime use and :meth:`fixed_for_tests`
    for deterministic unit tests.
    """

    user_agent: str = _DEFAULT_USER_AGENT
    """navigator.userAgent string."""

    build_id: str = _DEFAULT_BUILD_ID
    """chatgpt.com page data-build value."""

    script_src: str = _DEFAULT_SCRIPT_SRC
    """currentScript.src value."""

    language: str = "en"
    """navigator.language (primary)."""

    screen_width: int = 1366
    """screen.width."""

    screen_height: int = 1408
    """screen.height."""

    hardware_concurrency: int = 32
    """navigator.hardwareConcurrency."""

    js_heap_size_limit: int = 4_294_967_296
    """performance.memory.jsHeapSizeLimit."""

    navigator_probe: str = _DEFAULT_NAVIGATOR_PROBES[0]
    """``"X in navigator"`` probe value (format uses U+2212 MINUS SIGN separator)."""

    document_key: str = _DEFAULT_DOCUMENT_KEYS[0]
    """Random key from Object.keys(document)."""

    window_key: str = _DEFAULT_WINDOW_KEYS[0]
    """Random key from Object.getOwnPropertyNames(window)."""

    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """OAI-Device-Id / oai-did value (UUID)."""

    search_keys: str = ""
    """URLSearchParams(location.search) joined keys."""

    date_string: str = ""
    """new Date().toString(). Generated at construction when empty."""

    performance_now: float = 30412.5
    """performance.now() at token build time."""

    time_origin: float = 0.0
    """performance.timeOrigin (unix ms). Generated at construction when 0."""

    rand4: float = field(default_factory=random.random)
    """Slot [4] Math.random()."""

    window_probes: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    """Slots [18-24]: "X in window" probes."""

    def __post_init__(self) -> None:
        now = time.time()
        if not self.date_string:
            self.date_string = _format_browser_date(now, 480, "中国标准时间")
        if self.time_origin == 0.0:
            self.time_origin = now * 1000 - self.performance_now

    @classmethod
    def browser_default(cls) -> ConfigOptions:
        """Construct options with realistic runtime-sampled values."""
        return cls()

    @classmethod
    def fixed_for_tests(cls) -> ConfigOptions:
        """Construct fully deterministic options for unit tests.

        All wall-clock and random fields are fixed so tests produce stable
        results without asserting on variable values.
        """
        return cls(
            user_agent=_DEFAULT_USER_AGENT,
            build_id=_DEFAULT_BUILD_ID,
            script_src=_DEFAULT_SCRIPT_SRC,
            language="en",
            screen_width=1366,
            screen_height=1408,
            hardware_concurrency=32,
            js_heap_size_limit=4_294_967_296,
            navigator_probe=_DEFAULT_NAVIGATOR_PROBES[0],
            document_key=_DEFAULT_DOCUMENT_KEYS[0],
            window_key=_DEFAULT_WINDOW_KEYS[0],
            device_id="ee7b3426-19ed-4541-868a-ae24e57837ba",
            search_keys="",
            date_string="Tue Apr 21 2026 17:25:57 GMT+0800 (中国标准时间)",
            performance_now=30412.5,
            time_origin=1_776_763_524_501.3,
            rand4=0.12345,
            window_probes=(0, 0, 0, 0, 0, 0, 0),
        )


def _build_base_array(opts: ConfigOptions) -> list[Any]:
    """Build the 25-element config array.

    Slots [3] and [9] are set to their prepare-phase defaults (1 and a timing
    sample). The PoW solver overwrites [3] with the nonce and [9] with elapsed
    ms on each iteration.
    """
    return [
        str(opts.screen_width + opts.screen_height),  # [0]  string
        opts.date_string,  # [1]  string
        str(opts.js_heap_size_limit),  # [2]  string
        1,  # [3]  nonce placeholder (1 for prepare)
        opts.rand4,  # [4]  Math.random()
        opts.user_agent,  # [5]  navigator.userAgent
        opts.script_src,  # [6]  currentScript.src
        opts.build_id,  # [7]  data-build
        opts.language,  # [8]  navigator.language
        0,  # [9]  elapsed ms placeholder
        opts.navigator_probe,  # [10] "X in navigator" probe
        opts.document_key,  # [11] Object.keys(document)
        opts.window_key,  # [12] getOwnPropertyNames(window)
        opts.performance_now,  # [13] performance.now()
        opts.device_id,  # [14] device_id
        opts.search_keys,  # [15] URLSearchParams keys
        opts.hardware_concurrency,  # [16] hardwareConcurrency
        opts.time_origin,  # [17] performance.timeOrigin
        opts.window_probes[0],  # [18] "X in window" 1/7
        opts.window_probes[1],  # [19] "X in window" 2/7
        opts.window_probes[2],  # [20] "X in window" 3/7
        opts.window_probes[3],  # [21] "X in window" 4/7
        opts.window_probes[4],  # [22] "X in window" 5/7
        opts.window_probes[5],  # [23] "X in window" 6/7
        opts.window_probes[6],  # [24] "X in window" 7/7
    ]


def encode_config_array(config: list[Any]) -> str:
    """Encode a 25-element config array as standard base64(JSON.stringify(array))."""
    return base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode()


def encode_config(opts: ConfigOptions) -> str:
    """Build and encode the base config array from options."""
    return encode_config_array(_build_base_array(opts))


def build_requirements_token(opts: ConfigOptions | None = None) -> str:
    """Build the requirements p-token: ``gAAAAAC<base64(config)>``.

    Slot [3] is fixed to 1 (not the PoW nonce). Slot [9] carries
    ``performance_now``. Matches gproxy ``build_prepare_p``
    (prepare_p.rs:253-260): the requirements token has NO ``~S`` suffix — only
    the PoW answer (``gAAAAAB…~S``) from the solver does.
    """
    if opts is None:
        opts = ConfigOptions.browser_default()
    arr = _build_base_array(opts)
    arr[3] = 1
    arr[9] = opts.performance_now
    return f"gAAAAAC{encode_config_array(arr)}"


def build_prepare_p(opts: ConfigOptions | None = None) -> str:
    """Build the prepare-phase p value: ``gAAAAAC<base64(config)>``.

    Alias for :func:`build_requirements_token`. Both names are valid; the
    proof prefix (``gAAAAAB``) and ``~S`` suffix are used only on PoW solutions
    from the solver.
    """
    return build_requirements_token(opts=opts)
