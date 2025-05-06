import logging
import sys
import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple, TypedDict
import threading
import mido  # type: ignore[import]
import httpx
from soundfont_manager import SoundfontManager, FLUIDSYNTH_AVAILABLE

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


# Initialize the MCP server
mcp = FastMCP("midi_composition_server")

# Initialize the MIDI composition server
server = MidiCompositionServer()

# Create audio directory if it doesn't exist
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)


# MCP Interface
@mcp.tool()
async def create_project(
    name: str, tempo: int, time_signature: Tuple[int, int]
) -> Dict[str, Any]:
    """Create a new MIDI composition project.

    For a complete example of how to use the MIDI composition server, including:
    - Creating tracks with different instruments
    - Adding notes with proper timing and velocity
    - Exporting to MIDI and audio
    - Using MIDI settings for timing calculations

    Try running demonstrate_workflow() to see a working example.
    """
    success = server.create_project(name, tempo, time_signature)
    return {
        "success": success,
        "data": {
            "message": "Project created successfully. For a complete example of how to use the MIDI composition server, try running demonstrate_workflow()",
            "project_name": name,
            "tempo": tempo,
            "time_signature": time_signature,
        },
    }


@mcp.tool()
async def create_track(
    project_name: str, track_name: str, instrument: str, channel: int
) -> Dict[str, Any]:
    """Create a new track in the specified project.

    Args:
        project_name: Name of the project
        track_name: Name of the track
        instrument: Name of the instrument (must match a soundfont name from list_available_soundfonts())
        channel: MIDI channel number (0-15)

    Returns:
        Dict containing success status and error message if failed
    """
    if not server.load_project(project_name):
        return {"success": False, "error": f"Project '{project_name}' not found"}

    # Check if the instrument name matches an available soundfont
    try:
        with open("soundfontSources.json", "r") as f:
            available_soundfonts = json.load(f)
    except FileNotFoundError:
        return {"success": False, "error": "soundfontSources.json not found"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON in soundfontSources.json"}
    except Exception as e:
        return {"success": False, "error": f"Error reading soundfont sources: {str(e)}"}

    if instrument.lower() not in available_soundfonts:
        # Try to find a similar instrument name
        similar_instruments = [
            name
            for name in available_soundfonts.keys()
            if instrument.lower() in name or name in instrument.lower()
        ]
        suggestion = ""
        if similar_instruments:
            suggestion = (
                f" Did you mean one of these: {', '.join(similar_instruments)}?"
            )

        # Get list of valid instruments
        valid_instruments = get_valid_instruments()
        valid_instruments_list = "\nValid instruments are:\n" + "\n".join(
            f"- {name}" for name in valid_instruments
        )

        return {
            "success": False,
            "error": f"Invalid instrument name: {instrument}.{suggestion}{valid_instruments_list}\n\nUse list_available_soundfonts() to see available instruments.",
        }

    # Create the track with the instrument name
    success = server.create_track(track_name, instrument, channel)
    return {"success": success}


@mcp.tool()
async def mute_track(project_name: str, track_name: str) -> Dict[str, Any]:
    """Mute a track in the specified project."""
    if server.load_project(project_name):
        success = server.mute_track(track_name)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def solo_track(project_name: str, track_name: str) -> Dict[str, Any]:
    """Solo a track in the specified project."""
    if server.load_project(project_name):
        success = server.solo_track(track_name)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def set_track_volume(
    project_name: str, track_name: str, volume: float
) -> Dict[str, Any]:
    """Set the volume of a track in the specified project."""
    if server.load_project(project_name):
        success = server.set_track_volume(track_name, volume)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def set_track_pan(
    project_name: str, track_name: str, pan: float
) -> Dict[str, Any]:
    """Set the pan of a track in the specified project."""
    if server.load_project(project_name):
        success = server.set_track_pan(track_name, pan)
        return {"success": success}
    return {"success": False}


@mcp.tool()
async def get_project_info(project_name: str) -> Dict[str, Any]:
    """Get information about a project."""
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
    debug_info = server.debug_project_file(project_name)
    return {"success": True, "data": debug_info}


@mcp.tool()
async def list_projects() -> Dict[str, Any]:
    """List all available projects."""
    projects = [f.stem for f in server.workspace_dir.glob("*.json")]
    return {"success": True, "data": projects}


@mcp.tool()
async def export_midi(project_name: str, output_path: str) -> Dict[str, Any]:
    """Export a project as a MIDI file."""
    if server.load_project(project_name):
        success = server.export_midi(output_path)
        return {"success": success}
    return {"success": False}


def get_valid_instruments() -> List[str]:
    """Get a list of valid instrument names from soundfontSources.json."""
    try:
        with open("soundfontSources.json", "r") as f:
            sources = json.load(f)
        return sorted(sources.keys())
    except Exception:
        return []


@mcp.tool()
async def export_audio(
    project_name: str, output_path: str, format: str = "wav"
) -> Dict[str, Any]:
    """Export the current project to an audio file.

    Args:
        project_name: Name of the project to export
        output_path: Path where the audio file should be saved (will be saved in the audio directory)
        format: Audio format (wav, mp3, etc.)

    Returns:
        Dict containing success status and error message if failed
    """
    try:
        # Load the project if not already loaded
        if not server.current_project or server.current_project.name != project_name:
            load_result = server.load_project(project_name)
            if not load_result["success"]:
                return load_result

        # Ensure the output path is in the audio directory
        output_path = str(AUDIO_DIR / Path(output_path).name)

        # Export the audio
        result = server.export_audio(output_path, format)
        if result["success"]:
            return {
                "success": True,
                "message": f"Successfully exported audio to {output_path}",
                "file_path": output_path,
            }
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def play_audio_file(audio_path: str) -> Dict[str, Any]:
    """Play an audio file using the system's audio player.

    Args:
        audio_path: Path to the audio file to play

    Returns:
        Dict containing success status and error message if failed
    """
    try:
        result = server.play_audio_file(audio_path)
        if result["success"]:
            return {
                "success": True,
                "message": f"Successfully playing audio file: {audio_path}",
                "file_path": audio_path,
            }
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def add_notes(
    project_name: str, track_name: str, notes: List[Dict[str, int]]
) -> Dict[str, Any]:
    """Add multiple note events to a track in the specified project.

    Args:
        project_name: Name of the project
        track_name: Name of the track to add notes to
        notes: List of note events, each containing:
            - note: MIDI note number (0-127)
            - velocity: Note velocity (0-127)
            - time: Time in ticks (optional, defaults to 0)
            - duration: Duration in ticks (optional, defaults to 480 ticks = 1 beat)

    Returns:
        Dict with success status and error message if failed
    """
    if server.load_project(project_name):
        # Sort notes by time to ensure proper sequencing
        sorted_notes = sorted(notes, key=lambda x: x.get("time", 0))

        # Add notes in sequence
        for note in sorted_notes:
            # Ensure each note has required fields
            if "note" not in note or "velocity" not in note:
                return {
                    "success": False,
                    "error": "Each note must have 'note' and 'velocity' fields",
                }

            # Validate note values
            if not (0 <= note["note"] <= 127):
                return {
                    "success": False,
                    "error": f"Note value {note['note']} must be between 0 and 127",
                }
            if not (0 <= note["velocity"] <= 127):
                return {
                    "success": False,
                    "error": f"Velocity value {note['velocity']} must be between 0 and 127",
                }

            # Add the note to the track
            result = server.add_notes(track_name, [note])
            if not result["success"]:
                return result

        return {"success": True}
    return {"success": False, "error": f"Project '{project_name}' not found"}


@mcp.tool()
async def add_soundfont(soundfont_path: str) -> Dict[str, Any]:
    """Add a new soundfont to the server.

    Args:
        soundfont_path: Path to the .sf2 soundfont file

    Returns:
        Dict with success status and error message if failed
    """
    return server.add_soundfont(soundfont_path)


@mcp.tool()
async def list_soundfonts() -> Dict[str, Any]:
    """List all available soundfonts from soundfontSources.json.

    Returns:
        Dict containing success status and list of available soundfonts with their URLs
    """
    try:
        with open("soundfontSources.json", "r") as f:
            sources = json.load(f)

        soundfonts = [
            {"name": name.capitalize(), "url": url, "status": "available"}
            for name, url in sources.items()
        ]

        return {
            "success": True,
            "data": {"soundfonts": soundfonts, "count": len(soundfonts)},
        }
    except FileNotFoundError:
        return {"success": False, "error": "soundfontSources.json not found"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON in soundfontSources.json"}
    except Exception as e:
        return {"success": False, "error": f"Error listing soundfonts: {str(e)}"}


@mcp.tool()
async def remove_soundfont(soundfont_id: int) -> Dict[str, Any]:
    """Remove a soundfont from the server."""
    return server.remove_soundfont(soundfont_id)


@mcp.tool()
async def download_soundfont(search_term: str) -> Dict[str, Any]:
    """Search for and download a soundfont by name.

    The search_term must match one of the predefined soundfonts in soundfontSources.json.
    Available soundfonts are: piano, strings, bass, brass, choir, synth, organ, drums,
    flute, fx, pads, glockenspiel, and trumpet.

    Args:
        search_term: Name of the soundfont to download (must match a predefined soundfont)

    Returns:
        Dict containing success status and either the downloaded file path or error message
    """
    return server.download_soundfont(search_term)


@mcp.tool()
async def list_available_soundfonts() -> Dict[str, Any]:
    """List all available soundfonts that can be downloaded.

    This function reads soundfontSources.json and returns all available soundfonts.
    Each soundfont entry includes its name, category, and download URL.
    Use this to find available soundfonts before downloading them.

    Returns:
        Dict containing success status and list of available soundfonts with their categories
    """
    try:
        with open("soundfontSources.json", "r") as f:
            sources = json.load(f)

        soundfonts = [
            {"name": name.capitalize(), "category": name.capitalize(), "url": url}
            for name, url in sources.items()
        ]

        return {
            "success": True,
            "data": {"soundfonts": soundfonts, "count": len(soundfonts)},
        }
    except FileNotFoundError:
        return {"success": False, "error": "soundfontSources.json not found"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON in soundfontSources.json"}
    except Exception as e:
        return {"success": False, "error": f"Error listing soundfonts: {str(e)}"}


@mcp.tool()
async def demonstrate_workflow() -> Dict[str, Any]:
    """Demonstrate a typical workflow for using the MIDI composition server.

    This example shows:
    1. Creating a project with tempo and time signature
    2. Creating tracks with different instruments
    3. Adding notes with proper timing and velocity
    4. Exporting to MIDI and audio
    5. Using MIDI settings for timing calculations

    Returns:
        Dict containing the results of each step in the workflow
    """
    results = {}

    # Get MIDI settings first
    midi_settings = server.get_midi_settings()
    results["midi_settings"] = midi_settings
    ticks_per_beat = midi_settings["data"]["ticks_per_beat"]
    max_velocity = midi_settings["data"]["max_velocity"]

    # Create a new project
    project_result = server.create_project(
        name="Demo Project",
        tempo=120,  # 120 BPM
        time_signature=(4, 4),  # 4/4 time
    )
    results["create_project"] = project_result

    # Create tracks with different instruments
    tracks = [
        ("piano", "piano", 0),  # Piano on channel 0
        ("strings", "strings", 1),  # Strings on channel 1
        ("drums", "drums", 9),  # Drums on channel 9 (standard drum channel)
    ]

    for name, instrument, channel in tracks:
        track_result = server.create_track(name, instrument, channel)
        results[f"create_track_{name}"] = track_result

    # Add some notes to the piano track
    # Notes are in C major scale: C4, E4, G4
    piano_notes = [
        {
            "note": 60,  # C4
            "velocity": max_velocity // 2,  # Medium velocity
            "time": 0,  # Start at beginning
            "duration": ticks_per_beat,  # One beat duration
        },
        {
            "note": 64,  # E4
            "velocity": max_velocity // 2,
            "time": ticks_per_beat,  # Start after first note
            "duration": ticks_per_beat,
        },
        {
            "note": 67,  # G4
            "velocity": max_velocity // 2,
            "time": ticks_per_beat * 2,  # Start after second note
            "duration": ticks_per_beat,
        },
    ]
    piano_result = server.add_notes("piano", piano_notes)
    results["add_piano_notes"] = piano_result

    # Add some string notes (same notes, different timing)
    string_notes = [
        {
            "note": 60,  # C4
            "velocity": max_velocity // 3,  # Softer velocity
            "time": ticks_per_beat * 2,  # Start later
            "duration": ticks_per_beat * 2,  # Longer duration
        },
        {
            "note": 64,  # E4
            "velocity": max_velocity // 3,
            "time": ticks_per_beat * 4,
            "duration": ticks_per_beat * 2,
        },
        {
            "note": 67,  # G4
            "velocity": max_velocity // 3,
            "time": ticks_per_beat * 6,
            "duration": ticks_per_beat * 2,
        },
    ]
    string_result = server.add_notes("strings", string_notes)
    results["add_string_notes"] = string_result

    # Add some drum notes
    drum_notes = [
        {
            "note": 36,  # Bass drum
            "velocity": max_velocity,
            "time": 0,
            "duration": ticks_per_beat // 2,
        },
        {
            "note": 38,  # Snare drum
            "velocity": max_velocity,
            "time": ticks_per_beat,
            "duration": ticks_per_beat // 2,
        },
    ]
    drum_result = server.add_notes("drums", drum_notes)
    results["add_drum_notes"] = drum_result

    # Export to MIDI
    midi_path = "demo_project.mid"
    midi_result = server.export_midi(midi_path)
    results["export_midi"] = midi_result

    # Export to audio (if FluidSynth is available)
    if FLUIDSYNTH_AVAILABLE:
        audio_path = str(AUDIO_DIR / "demo_project.wav")
        audio_result = server.export_audio(audio_path, "wav")
        results["export_audio"] = audio_result

    return {
        "success": True,
        "data": {
            "results": results,
            "midi_file": midi_path,
            "audio_file": str(AUDIO_DIR / "demo_project.wav")
            if FLUIDSYNTH_AVAILABLE
            else None,
            "notes": {
                "piano": "C4, E4, G4 arpeggio",
                "strings": "C4, E4, G4 sustained chords",
                "drums": "Basic rock pattern (bass and snare)",
            },
        },
    }


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
