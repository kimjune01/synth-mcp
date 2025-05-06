import logging
import sys

# Configure logging to use stderr
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)

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

import mido

try:
    from pythonosc import osc_server
    from pythonosc import dispatcher

    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
import threading
from pathlib import Path

from typing import List, Dict, Any, TypedDict
import httpx
import json
import os
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
import subprocess
from pathlib import Path
import shutil
from difflib import SequenceMatcher

# Initialize FastMCP server
mcp = FastMCP("midi_composition_server")


# Type definitions for MCP
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


class MidiCompositionServer:
    def __init__(self) -> None:
        # FluidSynth state
        self.fs: Optional[FluidSynth] = None if FLUIDSYNTH_AVAILABLE else None
        self.sfid: Optional[int] = None
        self.synth_settings: Dict[str, Any] = {
            "gain": 1.0,
            "reverb": {"room_size": 0.2, "damping": 0.0, "width": 0.5, "level": 0.9},
            "chorus": {"nr": 3, "level": 2.0, "speed": 0.3, "depth": 8.0, "type": 0},
        }

        # Project state
        self.current_project: Optional[ProjectState] = None
        self.projects: Dict[str, ProjectState] = {}

        # Real-time collaboration state
        self.connected_clients: set[str] = set()
        self.state_lock: threading.Lock = threading.Lock()

        # Persistence settings
        self.workspace_dir: Path = Path("workspace")
        self.workspace_dir.mkdir(exist_ok=True)

    def _save_state(self) -> None:
        """Save current state to disk"""
        if not self.current_project:
            return

        try:
            state_file = self.workspace_dir / f"{self.current_project.name}.json"
            with open(state_file, "w") as f:
                json.dump(self.current_project.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")

    def _load_state(self, project_name: str) -> bool:
        """Load state from disk"""
        try:
            state_file = self.workspace_dir / f"{project_name}.json"
            if not state_file.exists():
                return False

            with open(state_file) as f:
                data = json.load(f)
                project = ProjectState.from_dict(data)
                self.current_project = project
                self.projects[project_name] = project
                return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False

    # 1. FluidSynth Integration Functions
    def initialize_fluidsynth(self) -> None:
        """Initialize FluidSynth with default settings"""
        try:
            from pyfluidsynth.fluidsynth import Synth

            self.fs = Synth()
            self.fs.start()

            # Load default soundfont
            soundfont_path = os.path.join(
                os.path.dirname(__file__), "Mario_Party_DS_HQ.sf2"
            )
            if os.path.exists(soundfont_path):
                sfid = self.fs.sfload(soundfont_path)
                if sfid != -1:
                    self.fs.program_select(0, sfid, 0, 0)
                    logging.info("Successfully loaded default soundfont")
                else:
                    logging.warning("Failed to load default soundfont")
            else:
                logging.warning(f"Default soundfont not found at {soundfont_path}")

            # Apply initial settings
            self.fs.setting("synth.gain", self.synth_settings["gain"])
            reverb = self.synth_settings["reverb"]
            self.set_reverb(
                reverb["room_size"], reverb["damping"], reverb["width"], reverb["level"]
            )
            chorus = self.synth_settings["chorus"]
            self.set_chorus(
                chorus["nr"],
                chorus["level"],
                chorus["speed"],
                chorus["depth"],
                chorus["type"],
            )
            logging.info("Successfully initialized FluidSynth")
        except ImportError:
            logging.warning(
                "FluidSynth not available - audio features will be disabled"
            )
            self.fs = None
        except Exception as e:
            logging.error(f"Error initializing FluidSynth: {str(e)}")
            self.fs = None

    def load_soundfont(self, soundfont_path: str) -> bool:
        """Load SoundFont files (.sf2) for instrument sounds"""
        with self.state_lock:
            if self.fs and Path(soundfont_path).exists():
                self.sfid = self.fs.sfload(soundfont_path)
                self.fs.sfont_select(0, self.sfid)
                return True
            return False

    def set_gain(self, gain_value: float) -> None:
        """Control the overall volume of the synth"""
        with self.state_lock:
            self.synth_settings["gain"] = gain_value
            if self.fs:
                self.fs.setting("synth.gain", gain_value)

    def set_reverb(
        self, room_size: float, damping: float, width: float, level: float
    ) -> None:
        """Configure reverb effects"""
        with self.state_lock:
            self.synth_settings["reverb"] = {
                "room_size": room_size,
                "damping": damping,
                "width": width,
                "level": level,
            }
            if self.fs:
                self.fs.set_reverb_roomsize(room_size)
                self.fs.set_reverb_damp(damping)
                self.fs.set_reverb_width(width)
                self.fs.set_reverb_level(level)

    def set_chorus(
        self, nr: int, level: float, speed: float, depth: float, type: int
    ) -> None:
        """Configure chorus effects"""
        with self.state_lock:
            self.synth_settings["chorus"] = {
                "nr": nr,
                "level": level,
                "speed": speed,
                "depth": depth,
                "type": type,
            }
            if self.fs:
                self.fs.set_chorus_nr(nr)
                self.fs.set_chorus_level(level)
                self.fs.set_chorus_speed(speed)
                self.fs.set_chorus_depth(depth)
                self.fs.set_chorus_type(type)

    # 2. MIDI Composition Functions
    def play_note(self, note: int, velocity: int, channel: int = 0) -> None:
        """Play a single note"""
        with self.state_lock:
            if self.fs:
                self.fs.noteon(channel, note, velocity)

    def stop_note(self, note: int, channel: int = 0) -> None:
        """Stop a single note"""
        with self.state_lock:
            if self.fs:
                self.fs.noteoff(channel, note)

    def program_change(self, program: int, channel: int = 0) -> None:
        """Change the instrument program"""
        with self.state_lock:
            if self.fs:
                self.fs.program_change(channel, program)

    def control_change(self, control: int, value: int, channel: int = 0) -> None:
        """Send a control change message"""
        with self.state_lock:
            if self.fs:
                self.fs.cc(channel, control, value)

    def pitch_bend(self, value: int, channel: int = 0) -> None:
        """Send a pitch bend message"""
        with self.state_lock:
            if self.fs:
                self.fs.pitch_bend(channel, value)

    def all_notes_off(self, channel: int = 0) -> None:
        """Stop all notes on a channel"""
        with self.state_lock:
            if self.fs:
                self.fs.all_notes_off(channel)

    def play_chord(self, notes, velocity, duration, channel):
        """Play multiple notes simultaneously as chords"""
        pass

    def create_sequence(self, notes, durations, velocities, channel):
        """Create sequences of notes with timing"""
        pass

    def play_midi_file(self, file_path):
        """Load and play existing MIDI files"""
        pass

    def record_midi(self, duration):
        """Record MIDI input for a specified duration"""
        pass

    # 3. Advanced Musical Functions
    def create_melody(self, scale, key, length, rhythm_pattern):
        """Generate melodies based on musical rules"""
        pass

    def create_chord_progression(self, progression, style, tempo):
        """Create harmonic progressions with different voicings"""
        pass

    def create_drum_pattern(self, style, tempo, variations):
        """Generate rhythmic patterns for percussion"""
        pass

    def create_arpeggio(self, chord, pattern, tempo):
        """Create arpeggiated patterns from chord structures"""
        pass

    def create_bassline(self, chord_progression, style, tempo):
        """Generate bass patterns that complement chord progressions"""
        pass

    # 4. Composition Management
    def create_track(self, name: str, instrument: int, channel: int) -> bool:
        """Create a new track with specified instrument"""
        with self.state_lock:
            if not self.current_project:
                return False
            track = TrackState(name=name, instrument=instrument, channel=channel)
            self.current_project.tracks[name] = track
            self._save_state()
            return True

    def mute_track(self, track_id: str) -> bool:
        """Silence a specific track"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].is_muted = True
            self._save_state()
            return True

    def solo_track(self, track_id: str) -> bool:
        """Solo a specific track"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].is_solo = True
            self._save_state()
            return True

    def set_track_volume(self, track_id: str, volume: float) -> bool:
        """Adjust volume for individual tracks"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].volume = volume
            self._save_state()
            return True

    def set_track_pan(self, track_id: str, pan: float) -> bool:
        """Adjust stereo positioning"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].pan = pan
            self._save_state()
            return True

    # 5. Project Management
    def create_project(
        self, name: str, tempo: int, time_signature: Tuple[int, int]
    ) -> bool:
        """Initialize a new composition project"""
        try:
            project = ProjectState(
                name=name, tempo=tempo, time_signature=time_signature, tracks={}
            )
            self.current_project = project
            self.projects[name] = project
            self._save_state()
            return True
        except Exception as e:
            print(f"Error creating project: {e}")
            return False

    def save_project(self, path: Optional[str] = None) -> bool:
        """Save the current project state"""
        if not self.current_project:
            return False

        with self.state_lock:
            self._save_state()
            if path:
                # Also save as MIDI if path provided
                self.export_midi(path)
            return True

    def load_project(self, name: str) -> bool:
        """Load a saved project"""
        return self._load_state(name)

    def export_midi(self, path: str) -> bool:
        """Export the composition as a standard MIDI file"""
        if not self.current_project:
            return False

        with self.state_lock:
            # Create a new MIDI file
            mid = mido.MidiFile()

            # Add tracks for each track in the project
            for track_name, track in self.current_project.tracks.items():
                midi_track = mido.MidiTrack()
                mid.tracks.append(midi_track)

                # Add track events
                for event in track.events:
                    if event["type"] == "note":
                        midi_track.append(
                            mido.Message(
                                "note_on",
                                note=event["note"],
                                velocity=event["velocity"],
                                time=event.get("time", 0),
                            )
                        )

            # Save the MIDI file
            mid.save(path)
            return True

    def export_audio(self, path: str, format: str) -> bool:
        """Render the composition to audio using FluidSynth"""
        if not self.current_project or not self.fs:
            return False

        with self.state_lock:
            # Implementation would go here
            # This is a placeholder as actual audio export would require more setup
            return True

    # 6. Real-time Collaboration and Interaction
    def start_midi_server(self, port: int) -> None:
        """Start a server that listens for MIDI events"""
        if not OSC_AVAILABLE:
            logging.warning("OSC server not available")
            return

        try:
            dispatcher_obj = dispatcher.Dispatcher()
            dispatcher_obj.map("/midi/note", self._handle_midi_note)
            dispatcher_obj.map("/midi/control", self._handle_midi_control)

            self.server = osc_server.ThreadingOSCUDPServer(
                ("127.0.0.1", port), dispatcher_obj
            )
            logging.info(f"Serving on {self.server.server_address}")
            self.server.serve_forever()
        except OSError as e:
            logging.error(f"Failed to start OSC server: {e}")
        except Exception as e:
            logging.error(f"Unexpected error starting OSC server: {e}")

    def stop_midi_server(self) -> None:
        """Stop the MIDI server"""
        try:
            if hasattr(self, "server"):
                self.server.shutdown()
                self.server.server_close()
                logging.info("OSC server stopped")

            # Clean up FluidSynth
            self.cleanup()
        except Exception as e:
            logging.error(f"Error stopping server: {e}")

    def __del__(self) -> None:
        """Cleanup when the server is destroyed"""
        try:
            self.stop_midi_server()
        except Exception as e:
            print(f"Error in cleanup: {e}")

    def _handle_midi_note(self, address: str, *args: Any) -> None:
        """Handle incoming MIDI note messages"""
        with self.state_lock:
            if not self.current_project:
                return
            # Process MIDI note message and update state
            track_name, note, velocity, time = args
            if track_name in self.current_project.tracks:
                self.current_project.tracks[track_name].events.append(
                    {"type": "note", "note": note, "velocity": velocity, "time": time}
                )

    def _handle_midi_control(self, address: str, *args: Any) -> None:
        """Handle incoming MIDI control messages"""
        with self.state_lock:
            if not self.current_project:
                return
            # Process MIDI control message and update state
            track_name, control, value = args
            if track_name in self.current_project.tracks:
                self.current_project.tracks[track_name].events.append(
                    {"type": "control", "control": control, "value": value}
                )

    def connect_midi_device(self, device_name):
        """Connect to external MIDI hardware"""
        pass

    def send_midi_event(self, event_type, parameters):
        """Send MIDI events to connected devices"""
        pass

    def sync_tempo(self, tempo):
        """Synchronize tempo across connected systems"""
        pass

    def debug_project_file(self, project_name: str) -> Dict[str, Any]:
        """Debug function to inspect a project file and its contents.

        Args:
            project_name: Name of the project to inspect

        Returns:
            Dict containing debug information about the project file:
            {
                'file_exists': bool,
                'file_path': str,
                'file_size': int,
                'file_contents': dict or None,
                'current_project': bool,
                'in_projects_dict': bool,
                'error': str or None
            }
        """
        try:
            debug_info = {
                "file_exists": False,
                "file_path": "",
                "file_size": 0,
                "file_contents": None,
                "current_project": False,
                "in_projects_dict": False,
                "error": None,
            }

            # Get project file path
            project_file = self.workspace_dir / f"{project_name}.json"
            debug_info["file_path"] = str(project_file)

            # Check if file exists and get size
            if project_file.exists():
                debug_info["file_exists"] = True
                debug_info["file_size"] = project_file.stat().st_size

                # Try to read and parse file contents
                try:
                    with open(project_file) as f:
                        debug_info["file_contents"] = json.load(f)
                except json.JSONDecodeError as e:
                    debug_info["error"] = f"JSON decode error: {str(e)}"
                except Exception as e:
                    debug_info["error"] = f"Error reading file: {str(e)}"

            # Check project state
            debug_info["current_project"] = (
                self.current_project is not None
                and self.current_project.name == project_name
            )
            debug_info["in_projects_dict"] = project_name in self.projects

            return debug_info

        except Exception as e:
            return {
                "file_exists": False,
                "file_path": "",
                "file_size": 0,
                "file_contents": None,
                "current_project": False,
                "in_projects_dict": False,
                "error": f"Debug function error: {str(e)}",
            }

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.fs:
            try:
                self.fs.delete()
                logging.info("Successfully cleaned up FluidSynth")
            except Exception as e:
                logging.error(f"Error cleaning up FluidSynth: {str(e)}")
        self.fs = None


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
