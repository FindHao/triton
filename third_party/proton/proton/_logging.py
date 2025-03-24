from torch.utils._traceback import CapturedTraceback
import logging
from typing import Optional, Callable, Any, Union
import os
import torch
import torch.distributed as dist
import torch._utils_internal
import torch._guards
import tempfile
import time
import json
log = logging.getLogger(__name__)
trace_log = logging.getLogger("torch.__trace")
triton_trace_log = logging.getLogger("torch.__triton_trace")
TRACE_ENV_VAR = "TORCH_TRACE"
TRITON_TRACE_HANDLER = None


class LazyTraceHandler(logging.StreamHandler):
    """Like FileHandler, but the file is allocated lazily only upon the first log message"""

    def __init__(self, root_dir: Optional[str]):
        # This is implemented in the same way that delay is implemented on
        # FileHandler
        self.root_dir = root_dir
        logging.Handler.__init__(self)
        self.stream = None
        self._builtin_open = open

    # cloned from FileHandler in cpython
    def close(self, append_text=None):
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
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                # Also see Issue #42378: we also rely on
                # self._closed being set to True there
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def emit(self, record):
        if self.stream is None:
            if self.root_dir is None:
                TRACE_LOG_DIR = "/logs"

                import torch.version as torch_version

                if (
                    hasattr(torch_version, "git_version")
                    and os.getenv("MAST_HPC_JOB_NAME") is None
                ):
                    log.info(
                        "LazyTraceHandler: disabled because not fbcode or conda on mast"
                    )
                elif not torch._utils_internal.justknobs_check("pytorch/trace:enable"):
                    log.info(
                        "LazyTraceHandler: disabled because justknobs_check('pytorch/trace:enable') returned False"
                    )
                elif not os.path.exists(TRACE_LOG_DIR):
                    log.info(
                        "LazyTraceHandler: disabled because %s does not exist",
                        TRACE_LOG_DIR,
                    )
                elif not os.access(TRACE_LOG_DIR, os.W_OK):
                    log.info(
                        "LazyTraceHandler: disabled because %s is not writeable",
                        TRACE_LOG_DIR,
                    )
                else:
                    self.root_dir = TRACE_LOG_DIR

            if self.root_dir is not None:
                os.makedirs(self.root_dir, exist_ok=True)
                ranksuffix = ""
                if dist.is_available() and dist.is_initialized():
                    ranksuffix = f"rank_{dist.get_rank()}_"
                self.stream = tempfile.NamedTemporaryFile(
                    mode="w+",
                    suffix=".log",
                    prefix=f"dedicated_log_torch_trace_{ranksuffix}",
                    dir=self.root_dir,
                    delete=False,
                )
                log.info("LazyTraceHandler: logging to %s", self.stream.name)
            else:
                # We go poof, remove and no-op
                trace_log.removeHandler(self)
                return
        if self.stream:
            super().emit(record)


class TritonLazyTraceHandler(LazyTraceHandler):
    """A specialized LazyTraceHandler for Triton compilation tracing that outputs JSON."""

    def __init__(
        self, root_dir: Optional[str], prefix="dedicated_log_torch_triton_trace_"
    ):
        super().__init__(root_dir)
        self.prefix = prefix
        self.stream = None

    def emit(self, record):
        # Create a new file for each emit call
        if self.root_dir is not None:
            os.makedirs(self.root_dir, exist_ok=True)
            ranksuffix = ""
            if dist.is_available() and dist.is_initialized():
                ranksuffix = f"rank_{dist.get_rank()}_"

            # Close previous file (if exists)
            if self.stream is not None:
                self.close(append_right_square_bracket=True)

            # Create a unique filename for each call using timestamp
            # Format: YYYYMMDD_HHMMSS
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = f"_{timestamp}"
            self.stream = tempfile.NamedTemporaryFile(
                mode="w+",
                suffix=".json",
                prefix=f"{self.prefix}{ranksuffix}{unique_id}_",
                dir=self.root_dir,
                delete=False,
            )

            # Write the beginning of the JSON array
            self.stream.write("[\n")
            self.first_record = True

            log.info("TritonLazyTraceHandler: logging to %s", self.stream.name)
        else:
            # If root_dir is not set, handle as in parent class
            trace_log.removeHandler(self)
            return

        if self.stream:
            # Add comma between records (except for the first one)
            if hasattr(self, "first_record") and self.first_record:
                self.first_record = False
            else:
                self.stream.write(",\n")

            # Format and write the record
            formatted = self.format(record)
            self.stream.write(formatted)
            self.flush()

    def close(self, append_text='\n]'):
        super().close(append_text)


class TritonJsonFormatter(logging.Formatter):
    """Format log records as JSON for Triton compilation tracing."""

    def format(self, record):
        # Create basic JSON structure
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
        }

        # Add metadata
        if hasattr(record, "metadata"):
            log_entry.update(record.metadata)

        # Add payload
        if hasattr(record, "payload") and record.payload is not None:
            # Try to parse JSON string, if fails add as raw string
            try:
                if isinstance(record.payload, str) and (
                    record.payload.startswith(
                        "{") or record.payload.startswith("[")
                ):
                    log_entry["payload"] = json.loads(record.payload)
                else:
                    log_entry["payload"] = record.payload
            except json.JSONDecodeError:
                log_entry["payload"] = record.payload

        # Handle exception information
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Convert the entire record to a JSON string
        return json.dumps(log_entry, indent=4)


def _init_triton_trace_logging(trace_dir_name=None):
    """Initialize the tracing log system for Triton compilation"""
    global TRITON_TRACE_HANDLER

    # If directory is set in environment variable, use it
    if trace_dir_name is None:
        trace_dir_name = os.environ.get(TRACE_ENV_VAR, None)

    # Ensure triton_trace_log doesn't propagate to parent loggers
    triton_trace_log.propagate = False
    triton_trace_log.setLevel(logging.DEBUG)

    # Clear existing handlers
    for handler in list(triton_trace_log.handlers):
        triton_trace_log.removeHandler(handler)

    # Create new handler
    if TRITON_TRACE_HANDLER is None:
        TRITON_TRACE_HANDLER = TritonLazyTraceHandler(trace_dir_name)

    # Set JSON formatter
    formatter = TritonJsonFormatter()
    TRITON_TRACE_HANDLER.setFormatter(formatter)

    # Add handler to logger
    triton_trace_log.addHandler(TRITON_TRACE_HANDLER)

    return triton_trace_log


def trace_structured_triton(
    name: str,
    metadata_fn: Callable[[], dict[str, Any]] = dict,
    *,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
    kernel_name: Optional[str] = None,
    grid_size: Optional[tuple] = None,
    compile_time_ms: Optional[float] = None,
    suppress_context: bool = False,
    new_file: bool = True,  # Whether to create a new log file for this call
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
    # Ensure triton trace logging system is initialized
    if not triton_trace_log.handlers:
        _init_triton_trace_logging()

    # If requesting to create a new file, reset the handler
    if new_file and TRITON_TRACE_HANDLER and TRITON_TRACE_HANDLER.stream is not None:
        # Close existing file
        TRITON_TRACE_HANDLER.close()
        # Ensure a new file is created on next emit
        TRITON_TRACE_HANDLER.stream = None

    record: dict[str, object] = {}
    metadata = metadata_fn()

    # Check if metadata is a tuple with string and int (source code and line number)
    if (
        isinstance(metadata, tuple)
        and len(metadata) == 2
        and isinstance(metadata[0], str)
        and isinstance(metadata[1], int)
        and name == "str"
    ):
        name = "sourcecode_line"

    record[name] = metadata
    if not suppress_context:
        # Get basic metadata

        record = {"event_type": name}
        # Add Triton-specific metadata
        if kernel_name is not None:
            record["kernel_name"] = kernel_name
        if grid_size is not None:
            record["grid_size"] = grid_size
        if compile_time_ms is not None:
            record["compile_time_ms"] = compile_time_ms

        if dist.is_available() and dist.is_initialized():
            record["rank"] = dist.get_rank()

        record["pid"] = os.getpid()

        trace_id = torch._guards.CompileContext.current_trace_id()
        if trace_id is not None:
            cid = trace_id.compile_id
            if cid is not None:
                if cid.compiled_autograd_id is not None:
                    record["compiled_autograd_id"] = cid.compiled_autograd_id
                if cid.frame_id is not None:
                    record["frame_id"] = cid.frame_id
                if cid.frame_compile_id is not None:
                    record["frame_compile_id"] = cid.frame_compile_id
            record["attempt"] = trace_id.attempt
        sourcecode_table: dict[str, int] = {}
        record["stack"] = []
        for frame in CapturedTraceback.extract(skip=1).summary():
            if frame.filename not in sourcecode_table:
                r = len(sourcecode_table)
                sourcecode_table[frame.filename] = r
            record["stack"].append(
                {
                    "line": frame.lineno,
                    "name": frame.name,
                    "filename": (frame.filename, sourcecode_table[frame.filename]),
                    "loc": frame.line,
                }
            )
    payload = payload_fn()
    triton_trace_log.debug("", extra={"metadata": record, "payload": payload})
