import logging
import sys
import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple, TypedDict
import threading
import mido

try:
    from pyfluidsynth.fluidsynth import Synth as FluidSynth

    logging.info("Successfully imported FluidSynth")
    FLUIDSYNTH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error importing FluidSynth: {e}")
    logging.warning(
        "FluidSynth features will be disabled. Audio export and real-time playback will not be available."
    )
    FLUIDSYNTH_AVAILABLE = False
except Exception as e:
    logging.error(f"Unexpected error importing FluidSynth: {e}")
    logging.warning(
        "FluidSynth features will be disabled. Audio export and real-time playback will not be available."
    )
    FLUIDSYNTH_AVAILABLE = False

try:
    from pythonosc import osc_server
    from pythonosc import dispatcher

    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

from mcp.server.fastmcp import FastMCP
from midi_server import MidiCompositionServer


# Type definitions
class ProjectInfo(TypedDict):
    name: str
    tempo: int
    time_signature: Tuple[int, int]
    tracks: Dict[str, Any]


class TrackInfo(TypedDict):
    name: str
    instrument: int
    channel: int
    is_muted: bool
    is_solo: bool
    volume: float
    pan: float


class DebugInfo(TypedDict):
    file_exists: bool
    file_path: str
    file_size: int
    file_contents: Optional[Dict[str, Any]]
    current_project: bool
    in_projects_dict: bool
    error: Optional[str]


@dataclass
class TrackState:
    name: str
    instrument: int
    channel: int
    is_muted: bool = False
    is_solo: bool = False
    volume: float = 1.0
    pan: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectState:
    name: str
    tempo: int
    time_signature: Tuple[int, int]
    tracks: Dict[str, TrackState]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectState":
        tracks = {k: TrackState(**v) for k, v in data.get("tracks", {}).items()}
        return cls(
            name=data["name"],
            tempo=data["tempo"],
            time_signature=tuple(data["time_signature"]),
            tracks=tracks,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "tracks": {k: asdict(v) for k, v in self.tracks.items()},
        }


# Initialize FastMCP server
mcp = FastMCP("midi_composition_server")


# MCP Interface
@mcp.tool()
async def create_project(
    name: str, tempo: int, time_signature: Tuple[int, int]
) -> Dict[str, Any]:
    """Create a new MIDI composition project."""
    server = MidiCompositionServer()
    success = server.create_project(name, tempo, time_signature)
    return {"success": success}


@mcp.tool()
async def create_track(
    project_name: str, track_name: str, instrument: int, channel: int
) -> Dict[str, Any]:
    """Create a new track in the specified project."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.create_track(track_name, instrument, channel)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def mute_track(project_name: str, track_name: str) -> Dict[str, Any]:
    """Mute a track in the specified project."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.mute_track(track_name)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def solo_track(project_name: str, track_name: str) -> Dict[str, Any]:
    """Solo a track in the specified project."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.solo_track(track_name)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def set_track_volume(
    project_name: str, track_name: str, volume: float
) -> Dict[str, Any]:
    """Set the volume of a track in the specified project."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.set_track_volume(track_name, volume)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def set_track_pan(
    project_name: str, track_name: str, pan: float
) -> Dict[str, Any]:
    """Set the pan of a track in the specified project."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.set_track_pan(track_name, pan)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def get_project_info(project_name: str) -> Dict[str, Any]:
    """Get information about a project."""
    server = MidiCompositionServer()
    if server.load_project(project_name) and server.current_project:
        project = server.current_project
        return {
            "success": True,
            "data": {
                "name": project.name,
                "tempo": project.tempo,
                "time_signature": project.time_signature,
                "tracks": {
                    name: asdict(track) for name, track in project.tracks.items()
                },
            },
        }
    return {"success": False}


@mcp.tool()
async def debug_project(project_name: str) -> Dict[str, Any]:
    """Get debug information about a project."""
    server = MidiCompositionServer()
    debug_info = server.debug_project_file(project_name)
    return {"success": True, "data": debug_info}


@mcp.tool()
async def list_projects() -> Dict[str, Any]:
    """List all available projects."""
    server = MidiCompositionServer()
    projects = [f.stem for f in server.workspace_dir.glob("*.json")]
    return {"success": True, "data": projects}


@mcp.tool()
async def export_midi(project_name: str, output_path: str) -> Dict[str, Any]:
    """Export a project as a MIDI file."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.export_midi(output_path)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def export_audio(
    project_name: str, output_path: str, format: str = "wav"
) -> Dict[str, Any]:
    """Export a project as an audio file."""
    server = MidiCompositionServer()
    if server.load_project(project_name):
        success = server.export_audio(output_path, format)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def add_note_event(
    project_name: str, track_name: str, note: int, velocity: int, time: int = 0
) -> Dict[str, Any]:
    """Add a note event to a track in the specified project."""
    server = MidiCompositionServer()
    if server.load_project(project_name) and server.current_project is not None:
        with server.state_lock:
            if track_name not in server.current_project.tracks:
                return {"success": False, "error": "Track not found"}

            track = server.current_project.tracks[track_name]
            track.events.append(
                {"type": "note", "note": note, "velocity": velocity, "time": time}
            )
            server._save_state()
            return {"success": True}
    return {"success": False}


@mcp.tool()
async def inspect_projects() -> Dict[str, Any]:
    """Inspect all projects in the workspace directory."""
    server = MidiCompositionServer()
    return server.inspect_projects()


def main() -> None:
    import threading
    import signal
    import sys
    import socket
    import json
    import time
    import select

    def signal_handler(sig, frame):
        logging.info("\nShutting down server...")
        server.stop_midi_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    server = MidiCompositionServer()
    server.initialize_fluidsynth()

    # Find an available port
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = find_free_port()
    logging.info(f"Starting server on port {port}")

    # Start the server in a background thread
    server_thread = threading.Thread(target=server.start_midi_server, args=(port,))
    server_thread.daemon = True
    server_thread.start()

    try:
        # Run the FastMCP server
        mcp.run()
    except KeyboardInterrupt:
        logging.info("\nShutting down server...")
        server.stop_midi_server()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        server.stop_midi_server()
    finally:
        server.stop_midi_server()


if __name__ == "__main__":
    main()
