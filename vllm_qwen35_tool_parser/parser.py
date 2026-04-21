# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom Qwen3.5 tool parser for vLLM.

Registered as --tool-call-parser qwen35_coder. Inherits from the stock
Qwen3CoderToolParser and overrides extract_tool_calls_streaming to ensure
schema-aware type coercion happens on every parameter value during
streaming — preventing array/object values from arriving at downstream
clients (Openclaw, Composio, etc.) as stringified JSON.

Module layout has shifted between vLLM versions, so imports are wrapped
in try/except to cover:
  - vLLM >= 0.11.x:  vllm.tool_parsers.*  (flat)
  - vLLM  < 0.11.x:  vllm.entrypoints.openai.tool_parsers.*  (nested)
"""
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.tool_parsers.qwen3coder_tool_parser import (
    Qwen3CoderToolParser,
)

logger = init_logger(__name__)

from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

@ToolParserManager.register_module("qwen35_coder")
class Qwen35CoderToolParser(Qwen3CoderToolParser):
    """Streaming override that forces schema-aware type coercion on every
    parameter so nested arrays/objects are emitted as real JSON values,
    not as stringified JSON inside the ``arguments`` object."""

    def __init__(self, tokenizer: TokenizerLike, tools=None):
        super().__init__(tokenizer, tools)

    @staticmethod
    def _coerce_to_native(value: str) -> Any:
        """If value is a self-contained JSON array or object, parse and return
        it as a native Python value. Otherwise return the original string.

        Only triggers when first *and* last non-whitespace chars are matching
        brackets, so plain strings (filenames, IDs, email addresses) are never
        touched. An inner json.loads() failure also falls through safely.
        """
        t = value.strip()
        if not t:
            return value
        if (t[0] == "[" and t[-1] == "]") or (t[0] == "{" and t[-1] == "}"):
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass
        return value

    def _convert_param_value(
        self,
        param_value: str,
        param_name: str,
        param_config: Any,
        function_name: str,
    ) -> Any:
        """Schema-aware conversion with heuristic fallback.

        Calls the parent's schema-driven _convert_param_value first. If the
        result is still a string, runs _coerce_to_native so that JSON
        arrays/objects in parameter bodies are promoted to native Python types
        before json.dumps() serializes them. This covers tools (e.g. Composio,
        MCP) whose JSON Schema omits explicit "type" annotations, which causes
        the schema lookup to miss and leaves values as strings — producing
        double-encoded output like "tools": "[{...}]" instead of "tools": [{...}].
        """
        converted = super()._convert_param_value(
            param_value, param_name, param_config, function_name
        )
        if isinstance(converted, str):
            converted = self._coerce_to_native(converted)
        return converted

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Update accumulated text
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                # Check if we're between tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            # No more tool calls to process yet
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # Add to prev_tool_call_arr immediately when we detect
                    # a tool call. This ensures finish_reason="tool_calls"
                    # even if parsing isn't complete.
                    already_added = any(
                        tool.get("name") == self.current_function_name
                        for tool in self.prev_tool_call_arr
                    )
                    if not already_added:
                        self.prev_tool_call_arr.append(
                            {
                                "name": self.current_function_name,
                                "arguments": "{}",  # Placeholder, will be updated later
                            }
                        )

                    # Send header with function info
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name, arguments=""
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # We've sent header, now handle function body
        if self.in_function:
            # Send opening brace if not sent yet
            if not self.json_started: ## and self.parameter_prefix not in delta_text:
                self.json_started = True
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            # Make sure json_started is set if we're processing parameters
            #if not self.json_started:
            #    self.json_started = True

            # Check for function end in accumulated text
            if not self.json_closed and self.function_end_token in tool_text:
                # Close JSON
                self.json_closed = True

                # Extract complete tool call to update prev_tool_call_arr
                # with final arguments.
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                            self.streaming_request.tools
                            if self.streaming_request
                            else None,
                        )
                        if parsed_tool:
                            # Update existing entry in
                            # prev_tool_call_arr with complete args
                            for i, tool in enumerate(self.prev_tool_call_arr):
                                if tool.get("name") == parsed_tool.function.name:
                                    args = parsed_tool.function.arguments
                                    self.prev_tool_call_arr[i]["arguments"] = args
                                    break
                    except Exception:
                        pass  # Ignore parsing errors during streaming

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )

                # Reset state for next tool
                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}

                return result

            # Look for parameters — find all parameter starts
            param_starts = []
            idx = 0
            while True:
                idx = tool_text.find(self.parameter_prefix, idx)
                if idx == -1:
                    break
                param_starts.append(idx)
                idx += len(self.parameter_prefix)

            # Check if we should start a new parameter
            if (
                not self.in_param
                and self.param_count < len(param_starts)
                and len(param_starts) > self.param_count
            ):
                # Process the next parameter
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" in remaining:
                    # We have the complete parameter name
                    name_end = remaining.find(">")
                    self.current_param_name = remaining[:name_end]

                    # Find the parameter value
                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    # Find where this parameter ends
                    param_end_idx = value_text.find(self.parameter_end_token)
                    if param_end_idx == -1:
                        # No closing tag, look for next parameter or
                        # function end
                        next_param_idx = value_text.find(self.parameter_prefix)
                        func_end_idx = value_text.find(self.function_end_token)

                        if next_param_idx != -1 and (
                            func_end_idx == -1 or next_param_idx < func_end_idx
                        ):
                            param_end_idx = next_param_idx
                        elif func_end_idx != -1:
                            param_end_idx = func_end_idx
                        else:
                            # Neither found, check if tool call is complete
                            if self.tool_call_end_token in tool_text:
                                # Tool call is complete, so parameter
                                # must be complete too — but guard against
                                # including </tool_call> in the value.
                                tool_end_in_value = value_text.find(
                                    self.tool_call_end_token
                                )
                                if tool_end_in_value != -1:
                                    param_end_idx = tool_end_in_value
                                else:
                                    param_end_idx = len(value_text)
                            else:
                                # Still streaming, wait for more content
                                return None

                    if param_end_idx != -1:
                        # Complete parameter found
                        param_value = value_text[:param_end_idx]
                        if param_value.endswith("\n"):
                            param_value = param_value[:-1]

                        # Store raw value for later processing
                        self.accumulated_params[self.current_param_name] = param_value

                        # Get parameter configuration for type conversion
                        param_config = self._get_arguments_config(
                            self.current_function_name or "",
                            self.streaming_request.tools
                            if self.streaming_request
                            else None,
                        )

                        # Convert param value to appropriate type per schema
                        converted_value = self._convert_param_value(
                            param_value,
                            self.current_param_name,
                            param_config,
                            self.current_function_name or "",
                        )

                        # Serialize the converted value as a JSON fragment
                        serialized_value = json.dumps(
                            converted_value, ensure_ascii=False
                        )

                        if self.param_count == 0:
                            json_fragment = (
                                f'"{self.current_param_name}": {serialized_value}'
                            )
                        else:
                            json_fragment = (
                                f', "{self.current_param_name}": {serialized_value}'
                            )

                        self.param_count += 1

                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    function=DeltaFunctionCall(arguments=json_fragment),
                                )
                            ]
                        )

            # Continue parameter value (unused in current impl since we
            # process complete parameters above, but retained for safety).
            if self.in_param:
                if self.parameter_end_token in delta_text:
                    # End of parameter
                    end_idx = delta_text.find(self.parameter_end_token)
                    value_chunk = delta_text[:end_idx]

                    # Skip past > if at start
                    if not self.current_param_value and ">" in value_chunk:
                        gt_idx = value_chunk.find(">")
                        value_chunk = value_chunk[gt_idx + 1:]

                    if not self.current_param_value and value_chunk.startswith("\n"):
                        value_chunk = value_chunk[1:]

                    # Store complete value
                    full_value = self.current_param_value + value_chunk
                    self.accumulated_params[self.current_param_name] = full_value

                    # Get parameter configuration for type conversion
                    param_config = self._get_arguments_config(
                        self.current_function_name or "",
                        self.streaming_request.tools
                        if self.streaming_request
                        else None,
                    )

                    # Convert the parameter value to the appropriate type
                    converted_value = self._convert_param_value(
                        full_value,
                        self.current_param_name or "",
                        param_config,
                        self.current_function_name or "",
                    )

                    # Serialize the converted value
                    serialized_value = json.dumps(converted_value, ensure_ascii=False)

                    # Since we've been streaming the quoted version,
                    # we need to close it properly
                    # This is complex - for now just complete the value
                    self.in_param = False
                    self.current_param_value = ""

                    # Just close the current parameter string
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments='"'),
                            )
                        ]
                    )
                else:
                    # Continue accumulating value
                    value_chunk = delta_text

                    # Handle first chunk after param name
                    if not self.current_param_value and ">" in value_chunk:
                        gt_idx = value_chunk.find(">")
                        value_chunk = value_chunk[gt_idx + 1:]

                    if not self.current_param_value and value_chunk.startswith("\n"):
                        value_chunk = value_chunk[1:]

                    if value_chunk:
                        # Stream the escaped delta
                        prev_escaped = (
                            json.dumps(self.current_param_value, ensure_ascii=False)[1:-1]
                            if self.current_param_value
                            else ""
                        )
                        self.current_param_value += value_chunk
                        full_escaped = json.dumps(
                            self.current_param_value, ensure_ascii=False
                        )[1:-1]
                        delta_escaped = full_escaped[len(prev_escaped):]

                        if delta_escaped:
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=delta_escaped
                                        ),
                                    )
                                ]
                            )

        return None
