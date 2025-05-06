import logging
import sys
import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    TypedDict,
    Union,
    cast,
    NotRequired,
    TypeVar,
    Protocol,
    Mapping,
)
import threading
import mido
import httpx
from soundfont_manager import SoundfontManager, FLUIDSYNTH_AVAILABLE

try:
    from pyfluidsynth.fluidsynth import Synth as FluidSynth

    logging.info("Successfully imported FluidSynth")
except ImportError as e:
    logging.warning(f"Error importing FluidSynth: {e}")
    logging.warning(
        "FluidSynth features will be disabled. Audio export and real-time playback will not be available."
    )
except Exception as e:
    logging.error(f"Unexpected error importing FluidSynth: {e}")
    logging.warning(
        "FluidSynth features will be disabled. Audio export and real-time playback will not be available."
    )

try:
    from pythonosc import osc_server
    from pythonosc import dispatcher

    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False


# Type definitions
class ProjectInfo(TypedDict):
    name: str
    tempo: int
    time_signature: Tuple[int, int]
    tracks: Dict[str, Any]


class TrackInfo(TypedDict):
    name: str
    instrument: str
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


class HasGet(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


T = TypeVar("T", bound=HasGet)


class NoteEvent(TypedDict, total=False):
    type: str  # "note"
    note: int
    velocity: int
    time: NotRequired[int]
    duration: NotRequired[int]


class ControlEvent(TypedDict, total=False):
    type: str  # "control"
    control: int
    value: int
    time: NotRequired[int]


Event = Union[NoteEvent, ControlEvent]


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


@dataclass
class TrackState:
    name: str
    instrument: str
    channel: int
    is_muted: bool = False
    is_solo: bool = False
    volume: float = 1.0
    pan: float = 0.0
    events: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Keep as Dict for flexibility


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


# General MIDI program numbers for instruments
INSTRUMENT_PROGRAMS = {
    "piano": 0,  # Acoustic Grand Piano
    "strings": 48,  # String Ensemble 1
    "bass": 32,  # Acoustic Bass
    "brass": 61,  # Brass Section
    "choir": 52,  # Choir Aahs
    "synth": 80,  # Square Lead
    "organ": 16,  # Rock Organ
    "drums": 0,  # Special case - use channel 9
    "flute": 73,  # Flute
    "fx": 96,  # FX 1 (rain)
    "pads": 88,  # Pad 1 (new age)
    "glockenspiel": 9,  # Glockenspiel
    "trumpet": 56,  # Trumpet
}

# MIDI settings
MIDI_TICKS_PER_BEAT = 480  # Standard MIDI resolution


class MidiCompositionServer:
    def __init__(self) -> None:
        """Initialize the MIDI composition server."""
        self.current_project: Optional[ProjectState] = None
        self.projects: Dict[str, ProjectState] = {}

        # Initialize soundfont manager
        self.soundfont_manager = SoundfontManager()

        # Initialize synth settings
        self.synth_settings = {
            "gain": 0.2,
            "reverb": {"room_size": 0.2, "damping": 0.0, "width": 0.5, "level": 0.9},
            "chorus": {"nr": 3, "level": 2.0, "speed": 0.3, "depth": 8.0, "type": 0},
        }

        # Initialize FluidSynth
        self.fs: Optional[FluidSynth] = None

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

    def play_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Play an audio file using afplay."""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                return {"success": False, "error": f"File not found: {audio_path}"}

            subprocess.run(["afplay", audio_path], check=True)
            return {"success": True}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"Error playing audio: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def create_track(self, name: str, instrument: str, channel: int) -> Dict[str, Any]:
        """Create a new track with specified instrument and ensure its soundfont is available.

        Args:
            name: Name of the track
            instrument: Name of the instrument (must match a soundfont name from soundfontSources.json)
            channel: MIDI channel number (0-15)

        Returns:
            Dict containing success status and error message if failed
        """
        with self.state_lock:
            if not self.current_project:
                return {"success": False, "error": "No project is currently loaded"}

            # Try to find and download the soundfont if not already available
            if not self.soundfont_manager.find_soundfont(instrument)["success"]:
                download_result = self.soundfont_manager.download_soundfont(instrument)
                if not download_result["success"]:
                    return {
                        "success": False,
                        "error": f"Could not find or download soundfont for instrument {instrument}",
                    }

            # Create the track
            track = TrackState(name=name, instrument=instrument, channel=channel)
            self.current_project.tracks[name] = track
            self._save_state()
            return {"success": True}

    def mute_track(self, track_id: str) -> Dict[str, Any]:
        """Silence a specific track.

        Args:
            track_id: Name of the track to mute

        Returns:
            Dict containing success status and error message if failed
        """
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return {"success": False, "error": f"Track '{track_id}' not found"}
            self.current_project.tracks[track_id].is_muted = True
            self._save_state()
            return {"success": True}

    def solo_track(self, track_id: str) -> Dict[str, Any]:
        """Solo a specific track.

        Args:
            track_id: Name of the track to solo

        Returns:
            Dict containing success status and error message if failed
        """
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return {"success": False, "error": f"Track '{track_id}' not found"}
            self.current_project.tracks[track_id].is_solo = True
            self._save_state()
            return {"success": True}

    def set_track_volume(self, track_id: str, volume: float) -> Dict[str, Any]:
        """Adjust volume for individual tracks.

        Args:
            track_id: Name of the track to adjust
            volume: Volume level (0.0 to 1.0)

        Returns:
            Dict containing success status and error message if failed
        """
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return {"success": False, "error": f"Track '{track_id}' not found"}
            self.current_project.tracks[track_id].volume = volume
            self._save_state()
            return {"success": True}

    def set_track_pan(self, track_id: str, pan: float) -> Dict[str, Any]:
        """Adjust stereo positioning.

        Args:
            track_id: Name of the track to adjust
            pan: Pan position (-1.0 to 1.0, where -1 is left, 0 is center, 1 is right)

        Returns:
            Dict containing success status and error message if failed
        """
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return {"success": False, "error": f"Track '{track_id}' not found"}
            self.current_project.tracks[track_id].pan = pan
            self._save_state()
            return {"success": True}

    def create_project(
        self, name: str, tempo: int, time_signature: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Initialize a new composition project.

        Args:
            name: Name of the project
            tempo: Tempo in BPM
            time_signature: Tuple of (numerator, denominator)

        Returns:
            Dict containing success status and error message if failed
        """
        try:
            project = ProjectState(
                name=name, tempo=tempo, time_signature=time_signature, tracks={}
            )
            self.current_project = project
            self.projects[name] = project
            self._save_state()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_project(self) -> Dict[str, Any]:
        """Save the current project state to a file.

        Returns:
            Dict containing success status and error message if failed
        """
        try:
            if not self.current_project:
                return {"success": False, "error": "No project is currently loaded"}

            # Save project state to JSON file
            project_file = self.workspace_dir / f"{self.current_project.name}.json"
            with open(project_file, "w") as f:
                json.dump(asdict(self.current_project), f, indent=2)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_project(self, name: str) -> Dict[str, Any]:
        """Load a project from a file.

        Args:
            name: Name of the project to load

        Returns:
            Dict containing success status and error message if failed
        """
        try:
            project_file = self.workspace_dir / f"{name}.json"
            if not project_file.exists():
                return {
                    "success": False,
                    "error": f"Project file not found: {name}.json",
                }

            with open(project_file) as f:
                data = json.load(f)
                self.current_project = ProjectState.from_dict(data)
                self.projects[name] = self.current_project
                return {"success": True}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Invalid JSON in project file: {name}.json",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_project(self, project_name: str) -> Dict[str, Any]:
        """Remove a project from the workspace."""
        try:
            # Get project file path
            project_file = self.workspace_dir / f"{project_name}.json"

            # Check if project exists
            if not project_file.exists():
                return {
                    "success": False,
                    "error": f"Project '{project_name}' not found",
                }

            # Remove from current project if it's the one being deleted
            if self.current_project and self.current_project.name == project_name:
                self.current_project = None

            # Remove from projects dict
            if project_name in self.projects:
                del self.projects[project_name]

            # Delete the project file
            project_file.unlink()

            return {"success": True}

        except Exception as e:
            logging.error(f"Error removing project {project_name}: {e}")
            return {"success": False, "error": str(e)}

    def export_midi(self, path: str) -> Dict[str, Any]:
        """Export the composition as a standard MIDI file.

        Args:
            path: Path to save the MIDI file

        Returns:
            Dict containing success status and error message if failed
        """
        if not self.current_project:
            return {"success": False, "error": "No project is currently loaded"}

        try:
            with self.state_lock:
                # Create a new MIDI file
                mid = mido.MidiFile()
                mid.ticks_per_beat = MIDI_TICKS_PER_BEAT

                # Add tracks for each track in the project
                for track_name, track in self.current_project.tracks.items():
                    midi_track = mido.MidiTrack()
                    mid.tracks.append(midi_track)

                    # Add program change message at the start of the track
                    # Get the program number for this instrument
                    program_num = INSTRUMENT_PROGRAMS.get(track.instrument.lower(), 0)

                    # Special handling for drums - use channel 9
                    if track.instrument.lower() == "drums":
                        track.channel = 9
                    else:
                        midi_track.append(
                            mido.Message(
                                "program_change",
                                program=program_num,
                                channel=track.channel,
                                time=0,
                            )
                        )

                    # Group events by time
                    events_by_time: Dict[int, List[Dict[str, Any]]] = {}
                    for event in track.events:
                        if isinstance(event, dict):
                            event_type = safe_get(event, "type")
                            if event_type == "note":
                                time = safe_get(event, "time", 0)
                                if time not in events_by_time:
                                    events_by_time[time] = []
                                events_by_time[time].append(event)

                    # Sort times and process events
                    for time in sorted(events_by_time.keys()):
                        # Add all note_on messages for this time
                        for event in events_by_time[time]:
                            note = safe_get(event, "note", 60)
                            velocity = safe_get(event, "velocity", 64)
                            duration = safe_get(event, "duration", MIDI_TICKS_PER_BEAT)

                            # Note on event
                            midi_track.append(
                                mido.Message(
                                    "note_on",
                                    note=note,
                                    velocity=velocity,
                                    time=0,  # Time is handled by the event grouping
                                    channel=track.channel,
                                )
                            )
                            # Note off event
                            midi_track.append(
                                mido.Message(
                                    "note_off",
                                    note=note,
                                    velocity=0,
                                    time=duration,
                                    channel=track.channel,
                                )
                            )

                # Save the MIDI file
                mid.save(path)
                return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def export_audio(self, path: str, format: str) -> Dict[str, Any]:
        """Render the composition to audio using FluidSynth.

        Args:
            path: Path to save the audio file
            format: Audio format (currently only 'wav' is supported)

        Returns:
            Dict containing success status and error message if failed
        """
        if not self.current_project:
            return {"success": False, "error": "No project is currently loaded"}

        if not FLUIDSYNTH_AVAILABLE:
            return {"success": False, "error": "FluidSynth is not available"}

        try:
            # First export to MIDI
            midi_path = path.replace(f".{format}", ".mid")
            midi_result = self.export_midi(midi_path)
            if not midi_result["success"]:
                return midi_result

            # Use FluidSynth to convert MIDI to WAV
            if format.lower() == "wav":
                # Get the current soundfont path
                soundfonts = self.soundfont_manager.list_soundfonts()
                if not soundfonts["success"] or not soundfonts["data"]["soundfonts"]:
                    # Use default soundfont
                    soundfont_path = "Mario_Party_DS_HQ.sf2"
                else:
                    soundfont_path = soundfonts["data"]["soundfonts"][0]["path"]

                cmd = [
                    "fluidsynth",
                    "-ni",
                    "-F",
                    path,
                    "-r",
                    "44100",
                    "-g",
                    "1.0",
                    soundfont_path,
                    midi_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                os.remove(midi_path)
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"FluidSynth error: {result.stderr}",
                    }
                return {"success": True}
            return {"success": False, "error": f"Unsupported format: {format}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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

    def debug_project_file(self, project_name: str) -> Dict[str, Any]:
        """Debug function to inspect a project file and its contents."""
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

    def inspect_projects(self) -> Dict[str, Any]:
        """Inspect all projects in the workspace directory."""
        projects = {}

        try:
            # Get all JSON files in the workspace directory
            project_files = list(self.workspace_dir.glob("*.json"))

            for project_file in project_files:
                try:
                    with open(project_file) as f:
                        data = json.load(f)
                        project_name = project_file.stem
                        projects[project_name] = {
                            "name": data.get("name", project_name),
                            "tempo": data.get("tempo", 120),
                            "time_signature": data.get("time_signature", [4, 4]),
                            "tracks": {
                                track_name: {
                                    "instrument": track_data.get("instrument", 0),
                                    "channel": track_data.get("channel", 0),
                                    "is_muted": track_data.get("is_muted", False),
                                    "is_solo": track_data.get("is_solo", False),
                                    "volume": track_data.get("volume", 1.0),
                                    "pan": track_data.get("pan", 0.0),
                                    "event_count": len(track_data.get("events", [])),
                                }
                                for track_name, track_data in data.get(
                                    "tracks", {}
                                ).items()
                            },
                        }
                except json.JSONDecodeError as e:
                    logging.error(f"Error reading project file {project_file}: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error processing {project_file}: {e}")
                    continue

            return {
                "success": True,
                "data": {"project_count": len(projects), "projects": projects},
            }
        except Exception as e:
            logging.error(f"Error inspecting projects: {e}")
            return {"success": False, "error": str(e)}

    def cleanup(self) -> None:
        """Clean up resources"""
        self.soundfont_manager.cleanup()

    def add_notes(self, track_name: str, notes: List[Dict[str, int]]) -> Dict[str, Any]:
        """Add notes to a track.

        Args:
            track_name: Name of the track to add notes to
            notes: List of note events with note, velocity, time, and duration

        Returns:
            Dict containing success status and error message if failed
        """
        if not self.current_project:
            return {"success": False, "error": "No project is currently loaded"}

        if track_name not in self.current_project.tracks:
            return {"success": False, "error": f"Track '{track_name}' not found"}

        try:
            track = self.current_project.tracks[track_name]
            for note in notes:
                # Check required fields
                if "note" not in note or "velocity" not in note:
                    return {
                        "success": False,
                        "error": "Note events must have 'note' and 'velocity' fields",
                    }

                # Validate MIDI note values
                note_value = note.get("note", 60)
                velocity = note.get("velocity", 64)

                if not (0 <= note_value <= 127):
                    return {
                        "success": False,
                        "error": f"Note value {note_value} must be between 0 and 127",
                    }
                if not (0 <= velocity <= 127):
                    return {
                        "success": False,
                        "error": f"Velocity value {velocity} must be between 0 and 127",
                    }

                # Create a properly typed NoteEvent
                event: NoteEvent = {
                    "type": "note",
                    "note": note_value,
                    "velocity": velocity,
                }
                if "time" in note:
                    event["time"] = note["time"]
                if "duration" in note:
                    event["duration"] = note["duration"]

                track.events.append(event)

            self._save_state()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_soundfont(self, soundfont_path: str) -> Dict[str, Any]:
        """Add a soundfont to FluidSynth."""
        return self.soundfont_manager.add_soundfont(soundfont_path)

    def remove_soundfont(self, soundfont_id: int) -> Dict[str, Any]:
        """Remove a soundfont from FluidSynth."""
        return self.soundfont_manager.remove_soundfont(soundfont_id)

    def list_soundfonts(self) -> Dict[str, Any]:
        """List all loaded soundfonts."""
        return self.soundfont_manager.list_soundfonts()

    def find_soundfont(self, search_term: str) -> Dict[str, Any]:
        """Search for a soundfont by name."""
        return self.soundfont_manager.find_soundfont(search_term)

    def download_soundfont(self, search_term: str) -> Dict[str, Any]:
        """Download a soundfont by name."""
        return self.soundfont_manager.download_soundfont(search_term)

    def get_midi_settings(self) -> Dict[str, Any]:
        """Get MIDI settings and constants.

        Returns:
            Dict containing MIDI settings including ticks per beat
        """
        return {
            "success": True,
            "data": {
                "ticks_per_beat": MIDI_TICKS_PER_BEAT,
                "max_note_value": 127,
                "max_velocity": 127,
                "max_channel": 15,
                "drum_channel": 9,
            },
        }
