import logging
import os
import json
import time
from collections import defaultdict
from typing import Optional, Callable, Any, Union
from torch.utils._traceback import CapturedTraceback
import torch.distributed as dist

# Configure basic logging
log = logging.getLogger(__name__)
triton_trace_log = logging.getLogger("triton.trace")
# Global variables
TRACE_ENV_VAR = "TORCH_TRACE"
TRITON_TRACE_HANDLER = None

torch_trace_folder = os.environ.get(TRACE_ENV_VAR, None)


def get_simplified_stack_trace(skip=1):
    """Get a simplified stack trace"""
    frames = []
    for frame in CapturedTraceback.extract(skip=skip).summary():
        frames.append({
            "line": frame.lineno,
            "name": frame.name,
            "filename": frame.filename,
            "loc": frame.line,
        })
    return frames


class TritonJsonFormatter(logging.Formatter):
    """Format log records as JSON for Triton compilation tracing."""

    def format(self, record):
        log_entry = record.metadata if hasattr(record, "metadata") else {}

        # Add timestamp
        log_entry["timestamp"] = self.formatTime(
            record, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Add payload if provided
        if hasattr(record, "payload") and record.payload is not None:
            try:
                if isinstance(record.payload, str) and (record.payload.startswith("{") or record.payload.startswith("[")):
                    log_entry["payload"] = json.loads(record.payload)
                else:
                    log_entry["payload"] = record.payload
            except json.JSONDecodeError:
                log_entry["payload"] = record.payload

        return json.dumps(log_entry, indent=2)


class TritonTraceHandler(logging.StreamHandler):
    """A handler for Triton compilation tracing that outputs JSON to separate files."""

    def __init__(self, root_dir: Optional[str], prefix="dedicated_log_triton_trace_"):
        logging.Handler.__init__(self)
        self.root_dir = root_dir
        self.prefix = prefix
        self.stream = None
        self.first_record = True

    def emit(self, record):
        # Create a new file for each emit call
        if self.root_dir is None:
            self.root_dir = os.environ.get(
                TRACE_ENV_VAR) or "/tmp/triton_traces"

        os.makedirs(self.root_dir, exist_ok=True)

        # Close previous file if exists
        if self.stream is not None:
            self.close()

        # Create unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rank_suffix = f"_rank{dist.get_rank()}" if dist.is_available(
        ) and dist.is_initialized() else ""
        filename = f"{self.prefix}{timestamp}{rank_suffix}.json"

        self.stream = open(os.path.join(self.root_dir, filename), "w")
        self.stream.write("[\n")  # Start JSON array
        self.first_record = True

        # Format and write record
        if self.first_record:
            self.first_record = False
        else:
            self.stream.write(",\n")

        formatted = self.format(record)
        self.stream.write(formatted)
        self.flush()

    def close(self, append_text="\n]"):
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        if append_text:
                            self.stream.write(append_text)
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # PyTorch Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                # Also see PyTorch Issue #42378: we also rely on
                # self._closed being set to True there
                logging.StreamHandler.close(self)
        finally:
            self.release()


def trace_structured_triton(
    name: str,
    metadata_fn: Callable[[], dict[str, Any]] = dict,
    *,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
    kernel_name: Optional[str] = None,
    grid_size: Optional[tuple] = None,
    compile_time_ms: Optional[float] = None,
    new_file: bool = True,
):
    """
    Record structured trace information for Triton kernel compilation

    Args:
        name: Name of the trace event
        metadata_fn: Function that returns metadata dictionary
        payload_fn: Function that returns payload data
        kernel_name: Triton kernel name
        grid_size: Grid size
        compile_time_ms: Compile time in milliseconds
        new_file: Whether to create a new log file for this call
    """
    global TRITON_TRACE_HANDLER

    # Initialize logging if needed
    if not triton_trace_log.handlers or TRITON_TRACE_HANDLER is None:
        trace_dir = os.environ.get(TRACE_ENV_VAR)
        triton_trace_log.setLevel(logging.DEBUG)
        triton_trace_log.propagate = False

        # Clear existing handlers
        for handler in list(triton_trace_log.handlers):
            triton_trace_log.removeHandler(handler)

        TRITON_TRACE_HANDLER = TritonTraceHandler(trace_dir)
        formatter = TritonJsonFormatter()
        TRITON_TRACE_HANDLER.setFormatter(formatter)
        triton_trace_log.addHandler(TRITON_TRACE_HANDLER)

    # Create new file if requested
    if new_file and TRITON_TRACE_HANDLER and TRITON_TRACE_HANDLER.stream is not None:
        TRITON_TRACE_HANDLER.close()
        TRITON_TRACE_HANDLER.stream = None

    # Prepare the record
    record = {"event_type": name}

    # Add Triton-specific metadata
    if kernel_name:
        record["kernel_name"] = kernel_name
    if grid_size:
        record["grid_size"] = grid_size
    if compile_time_ms:
        record["compile_time_ms"] = compile_time_ms

    # Add basic context information
    record["pid"] = os.getpid()
    if dist.is_available() and dist.is_initialized():
        record["rank"] = dist.get_rank()

    # Add custom metadata
    custom_metadata = metadata_fn()
    if custom_metadata:
        record.update(custom_metadata)

    # Get stack trace
    record["stack"] = get_simplified_stack_trace(skip=2)

    # Log the record
    payload = payload_fn()
    triton_trace_log.debug("", extra={"metadata": record, "payload": payload})


def extract_python_source_info(trace_data, source, is_ir_source):
    """
    Extract Python source code information from the source object and add it to trace_data.

    Args:
        trace_data: Dictionary to store extracted information
        source: Source object (ASTSource or IRSource)
        is_ir_source: Whether the source is an IR source
    """
    if is_ir_source or not hasattr(source, 'fn'):
        return

    import inspect
    try:
        # Get the original Python source code for the kernel
        target_fn = source.fn.fn
        python_source_file = inspect.getfile(target_fn)
        start_line_number = inspect.getsourcelines(target_fn)[1]
        end_line_number = start_line_number + \
            len(inspect.getsourcelines(target_fn)[0])

        trace_data["python_source_file_path"] = python_source_file
        trace_data["python_source_start_line_number"] = start_line_number
        trace_data["python_source_end_line_number"] = end_line_number

        # Get function source code directly using inspect.getsource()
        trace_data["python_source_code"] = inspect.getsource(target_fn)
    except (TypeError, OSError) as e:
        trace_data["python_source_error"] = str(e)


def extract_file_content(trace_data, metadata_group):
    """
    Extract file content from metadata_group and add it to trace_data.

    Args:
        trace_data: Dictionary to store extracted information
        metadata_group: Dictionary mapping filenames to file paths
    """
    for ir_filename, file_path in metadata_group.items():
        # Add file path to trace data
        trace_data["file_path"][ir_filename] = file_path

        # Read and add file content for text-based IR files
        text_file_extensions = ['.ttir', '.ttgir', '.llir', '.ptx', '.json']
        if any(ir_filename.endswith(ext) for ext in text_file_extensions):
            try:
                with open(file_path, 'r') as f:
                    trace_data["file_content"][ir_filename] = f.read()
            except (IOError, UnicodeDecodeError):
                # Skip if file can't be read as text
                pass


def maybe_trace_triton(metadata_path, metadata_group, src, ir_source):
    trace_data = defaultdict(dict)
    if not torch_trace_folder:
        return trace_data
    if metadata_path is not None:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            for key, value in metadata.items():
                trace_data["metadata"][key] = value
    if metadata_group:
        extract_file_content(trace_data, metadata_group)
    if src:
        extract_python_source_info(trace_data, src, ir_source)

    trace_structured_triton(
        "triton.kernel",
        payload_fn=lambda: json.dumps(trace_data),
    )
    return trace_data
