"""Tests for ccproxy utilities."""

import io
import json
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

import ccproxy.utils as utils_mod
from ccproxy.utils import calculate_duration_ms, get_template_file, get_templates_dir, parse_session_id


class TestGetTemplatesDir:
    def test_templates_dir_package_layout(self, tmp_path: Path) -> None:
        """Test finding templates adjacent to the package module."""
        src_dir = tmp_path / "src" / "ccproxy"
        src_dir.mkdir(parents=True)
        utils_file = src_dir / "utils.py"
        utils_file.touch()

        templates_dir = src_dir / "templates"
        templates_dir.mkdir()
        (templates_dir / "ccproxy.yaml").touch()

        with patch("ccproxy.utils.__file__", str(utils_file)):
            result = get_templates_dir()
            assert result == templates_dir

    def test_templates_dir_installed_mode(self, tmp_path: Path) -> None:
        """Test finding templates in installed package mode."""
        # Create a fake module location
        fake_module = tmp_path / "fake" / "location" / "ccproxy"
        fake_module.mkdir(parents=True)
        fake_utils = fake_module / "utils.py"
        fake_utils.touch()

        # Create templates inside the package
        templates_dir = fake_module / "templates"
        templates_dir.mkdir()
        (templates_dir / "ccproxy.yaml").touch()

        # Mock __file__
        with patch("ccproxy.utils.__file__", str(fake_utils)):
            result = get_templates_dir()
            assert result == templates_dir

    def test_templates_dir_not_found(self) -> None:
        """Test error when templates directory not found."""
        # Mock __file__ to point to a location without templates
        with (
            patch("ccproxy.utils.__file__", "/nowhere/utils.py"),
            patch.object(Path, "exists", return_value=False),
            pytest.raises(RuntimeError) as exc_info,
        ):
            get_templates_dir()

        assert "Could not find templates directory" in str(exc_info.value)


class TestGetTemplateFile:
    @patch("ccproxy.utils.get_templates_dir")
    def test_get_existing_template(self, mock_get_templates: Mock, tmp_path: Path) -> None:
        """Test getting an existing template file."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        template_file = templates_dir / "test.yaml"
        template_file.write_text("test content")

        mock_get_templates.return_value = templates_dir

        result = get_template_file("test.yaml")
        assert result == template_file

    @patch("ccproxy.utils.get_templates_dir")
    def test_get_nonexistent_template(self, mock_get_templates: Mock, tmp_path: Path) -> None:
        """Test error when template file doesn't exist."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        mock_get_templates.return_value = templates_dir

        with pytest.raises(FileNotFoundError) as exc_info:
            get_template_file("missing.yaml")

        assert "Template file not found: missing.yaml" in str(exc_info.value)


class TestCalculateDurationMs:
    def test_calculate_duration_with_floats(self) -> None:
        """Test duration calculation with float timestamps."""
        start_time = 1000.0
        end_time = 1002.5

        result = calculate_duration_ms(start_time, end_time)

        assert result == 2500.0  # 2.5 seconds = 2500 ms

    def test_calculate_duration_with_timedelta(self) -> None:
        """Test duration calculation with timedelta objects."""
        start_time = timedelta(seconds=0)
        end_time = timedelta(seconds=1, milliseconds=500)

        result = calculate_duration_ms(start_time, end_time)

        assert result == 1500.0  # 1.5 seconds = 1500 ms

    def test_calculate_duration_with_mixed_types(self) -> None:
        """Test that mixed types are handled gracefully."""
        # Mixed types that don't support subtraction should return 0.0
        start_time = 0
        end_time = timedelta(seconds=2)

        # This will fail because int - timedelta is not supported
        result = calculate_duration_ms(start_time, end_time)

        # Should return 0.0 due to TypeError
        assert result == 0.0

    def test_calculate_duration_with_invalid_types(self) -> None:
        """Test that invalid types return 0.0."""
        # String types should cause TypeError
        result = calculate_duration_ms("start", "end")
        assert result == 0.0

        # None types should cause TypeError
        result = calculate_duration_ms(None, None)
        assert result == 0.0

        # Object without subtraction support
        result = calculate_duration_ms({"time": 1}, {"time": 2})
        assert result == 0.0

    def test_calculate_duration_rounding(self) -> None:
        """Test that results are rounded to 2 decimal places."""
        start_time = 1000.0
        end_time = 1000.0012345

        result = calculate_duration_ms(start_time, end_time)

        assert result == 1.23  # Should be rounded to 2 decimal places

    def test_calculate_duration_negative(self) -> None:
        """Test calculation when end time is before start time."""
        start_time = 2000.0
        end_time = 1000.0

        result = calculate_duration_ms(start_time, end_time)

        assert result == -1000000.0  # Negative duration is allowed


class TestFindAvailablePort:
    """Tests for find_available_port function."""

    def test_returns_a_port_in_range(self) -> None:
        from ccproxy.utils import find_available_port

        port = find_available_port()
        assert 1 <= port <= 65535

    def test_returned_port_is_bindable(self) -> None:
        import socket

        from ccproxy.utils import find_available_port

        port = find_available_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    def test_bind_failure_propagates(self) -> None:
        from ccproxy.utils import find_available_port

        with patch("socket.socket") as mock_sock_cls, pytest.raises(OSError, match="bind failed"):
            mock_sock = mock_sock_cls.return_value.__enter__.return_value
            mock_sock.bind.side_effect = OSError("bind failed")
            find_available_port()


class TestFormatValue:
    """Tests for _format_value helper."""

    def test_string_truncation(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value("x" * 100, max_width=10)
        assert "..." in result

    def test_object_truncation(self) -> None:
        from ccproxy.utils import _format_value

        class Big:
            def __str__(self) -> str:
                return "x" * 100

        result = _format_value(Big(), max_width=10)
        assert "..." in result

    def test_string_escapes_markup(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value("[bold]text[/bold]")
        assert r"\[" in result


class TestParseSessionId:
    """Tests for parse_session_id."""

    def test_json_format(self) -> None:
        user_id = json.dumps({"device_id": "dev1", "account_uuid": "acc1", "session_id": "abc123"})
        assert parse_session_id(user_id) == "abc123"

    def test_json_format_minimal(self) -> None:
        user_id = json.dumps({"session_id": "xyz"})
        assert parse_session_id(user_id) == "xyz"

    def test_json_format_no_session_id(self) -> None:
        user_id = json.dumps({"device_id": "dev1"})
        assert parse_session_id(user_id) is None

    def test_json_format_empty_session_id(self) -> None:
        user_id = json.dumps({"session_id": ""})
        assert parse_session_id(user_id) is None

    def test_json_format_invalid_json(self) -> None:
        assert parse_session_id("{not valid json") is None

    def test_legacy_format(self) -> None:
        assert parse_session_id("user_hash_account_uuid_session_sid123") == "sid123"

    def test_legacy_format_multiple_session_separators(self) -> None:
        assert parse_session_id("a_session_b_session_c") is None

    def test_neither_format(self) -> None:
        assert parse_session_id("plain-user-id") is None

    def test_empty_string(self) -> None:
        assert parse_session_id("") is None


# ---------------------------------------------------------------------------
# Helpers shared by the debug-table tests below
# ---------------------------------------------------------------------------


def _capture(fn: Any, *args: Any, **kwargs: Any) -> str:
    """Run fn(*args, **kwargs) with the module-level console redirected to a buffer."""
    buf = io.StringIO()
    original = utils_mod.console
    utils_mod.console = Console(file=buf, no_color=True, width=160)
    try:
        fn(*args, **kwargs)
    finally:
        utils_mod.console = original
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _format_value — remaining branches
# ---------------------------------------------------------------------------


class TestFormatValueAllBranches:
    def test_none_returns_dim_markup(self) -> None:
        from ccproxy.utils import _format_value

        assert _format_value(None) == "[dim]None[/dim]"

    def test_true_returns_green_markup(self) -> None:
        from ccproxy.utils import _format_value

        assert _format_value(True) == "[green]True[/green]"

    def test_false_returns_red_markup(self) -> None:
        from ccproxy.utils import _format_value

        assert _format_value(False) == "[red]False[/red]"

    def test_int_returns_cyan_markup(self) -> None:
        from ccproxy.utils import _format_value

        assert _format_value(42) == "[cyan]42[/cyan]"

    def test_float_returns_cyan_markup(self) -> None:
        from ccproxy.utils import _format_value

        assert _format_value(3.14) == "[cyan]3.14[/cyan]"

    def test_list_returns_dim_with_length(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value([1, 2, 3])
        assert "list" in result
        assert "3" in result

    def test_tuple_returns_dim_with_length(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value((10, 20))
        assert "tuple" in result
        assert "2" in result

    def test_dict_returns_dim_with_length(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value({"a": 1, "b": 2})
        assert "dict" in result
        assert "2" in result

    def test_callable_returns_magenta_with_parens(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value(len)
        assert "len()" in result

    def test_arbitrary_object_str_representation(self) -> None:
        from ccproxy.utils import _format_value

        class Obj:
            def __str__(self) -> str:
                return "my_obj"

        assert "my_obj" in _format_value(Obj())

    def test_object_truncation_with_max_width(self) -> None:
        from ccproxy.utils import _format_value

        class Big:
            def __str__(self) -> str:
                return "x" * 100

        result = _format_value(Big(), max_width=10)
        assert "..." in result
        assert len(result) <= 13  # 10 - 3 + "..." + possible markup

    def test_no_max_width_no_truncation(self) -> None:
        from ccproxy.utils import _format_value

        long_str = "y" * 200
        result = _format_value(long_str)
        assert "..." not in result

    def test_string_with_markup_chars_escaped(self) -> None:
        from ccproxy.utils import _format_value

        result = _format_value("[bold]text")
        assert r"\[" in result


# ---------------------------------------------------------------------------
# debug_table and its dispatch branches
# ---------------------------------------------------------------------------


class TestDebugTableDispatch:
    def test_dict_input_prints_keys(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, {"alpha": 1, "beta": "two"})
        assert "alpha" in output
        assert "beta" in output

    def test_dict_with_title_prints_title(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, {"k": "v"}, title="MyTitle")
        assert "MyTitle" in output

    def test_list_input_prints_indices(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, [10, 20, 30])
        assert "0" in output
        assert "1" in output
        assert "2" in output

    def test_tuple_input_prints_indices(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, (100, 200))
        assert "0" in output

    def test_object_with_dict_prints_attributes(self) -> None:
        from ccproxy.utils import debug_table

        class Obj:
            def __init__(self) -> None:
                self.x = 1
                self.y = "hello"

        output = _capture(debug_table, Obj())
        assert "x" in output
        assert "y" in output

    def test_object_with_show_methods_includes_callable(self) -> None:
        from ccproxy.utils import debug_table

        class Obj:
            def greet(self) -> str:
                return "hi"

        output = _capture(debug_table, Obj(), show_methods=True)
        assert "greet" in output

    def test_fallback_bare_value_printed(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, 42)
        assert "42" in output

    def test_compact_false_branch_dict(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, {"k": "v"}, compact=False)
        assert "k" in output

    def test_compact_false_branch_list(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, [1, 2], compact=False)
        assert "0" in output

    def test_max_width_passed_to_print_dict(self) -> None:
        from ccproxy.utils import debug_table

        output = _capture(debug_table, {"k": "x" * 200}, max_width=20)
        assert "k" in output


# ---------------------------------------------------------------------------
# dt (alias)
# ---------------------------------------------------------------------------


class TestDt:
    def test_dt_is_alias_for_debug_table(self) -> None:
        from ccproxy.utils import dt

        output = _capture(dt, {"key": "value"})
        assert "key" in output


# ---------------------------------------------------------------------------
# dv — debug variables
# ---------------------------------------------------------------------------


class TestDv:
    def test_positional_args_printed(self) -> None:
        from ccproxy.utils import dv

        output = _capture(dv, 42, "hello")
        assert "42" in output
        assert "hello" in output

    def test_kwargs_printed(self) -> None:
        from ccproxy.utils import dv

        output = _capture(dv, count=5, name="test")
        assert "count" in output
        assert "name" in output
        assert "test" in output

    def test_mixed_positional_and_kwargs(self) -> None:
        from ccproxy.utils import dv

        output = _capture(dv, 99, label="foo")
        assert "99" in output
        assert "label" in output
        assert "foo" in output


# ---------------------------------------------------------------------------
# d — ultra-compact debug print
# ---------------------------------------------------------------------------


class TestD:
    def test_dict_dispatches_to_debug_table(self) -> None:
        from ccproxy.utils import d

        output = _capture(d, {"k": "v"})
        assert "k" in output

    def test_with_width_parameter(self) -> None:
        from ccproxy.utils import d

        output = _capture(d, {"k": "x" * 50}, w=20)
        assert "k" in output


# ---------------------------------------------------------------------------
# p — minimal compact debug print
# ---------------------------------------------------------------------------


class TestP:
    def test_dict_prints_key_value(self) -> None:
        from ccproxy.utils import p

        output = _capture(p, {"mykey": "myval"})
        assert "mykey" in output

    def test_list_prints_indices(self) -> None:
        from ccproxy.utils import p

        output = _capture(p, [10, 20])
        assert "0" in output

    def test_tuple_prints_indices(self) -> None:
        from ccproxy.utils import p

        output = _capture(p, (30, 40))
        assert "0" in output

    def test_object_with_dict_prints_attrs(self) -> None:
        from ccproxy.utils import p

        class Obj:
            def __init__(self) -> None:
                self.public_attr = "exposed"
                self._private = "hidden"

        output = _capture(p, Obj())
        assert "public_attr" in output
        assert "_private" not in output

    def test_fallback_bare_value_printed(self) -> None:
        from ccproxy.utils import p

        # A bare int has no __dict__ — falls to console.print(obj)
        output = _capture(p, 12345)
        assert "12345" in output


class TestPrintObjectEdgeCases:
    def test_callable_attributes_skipped_when_show_methods_false(self) -> None:
        from ccproxy.utils import _print_object

        class ObjWithMethod:
            data = "value"

            def compute(self) -> int:
                return 42

        output = _capture(_print_object, ObjWithMethod(), "Title", None, False, True)
        assert "data" in output
        assert "compute" not in output

    def test_attribute_raising_exception_stored_as_placeholder(self) -> None:
        from ccproxy.utils import _print_object

        class Tricky:
            @property
            def explodes(self) -> str:
                raise RuntimeError("access denied")

            normal = "ok"

        output = _capture(_print_object, Tricky(), "Tricky", None, False, True)
        assert "normal" in output
        assert "explodes" in output
        assert "unable to access" in output
