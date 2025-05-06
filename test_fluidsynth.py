import os

os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"

try:
    from pyfluidsynth.fluidsynth import Synth

    print("Successfully imported FluidSynth")

    # Try to create a synth instance
    fs = Synth()
    print("Successfully created FluidSynth instance")

    # Try to start the synth
    fs.start()
    print("Successfully started FluidSynth")

    # Try to load a soundfont
    soundfont_path = "soundfonts/GeneralUser_GS.sf2"
    if os.path.exists(soundfont_path):
        sfid = fs.sfload(soundfont_path)
        fs.sfont_select(0, sfid)
        print(f"Successfully loaded soundfont: {soundfont_path}")
    else:
        print(f"Soundfont not found at: {soundfont_path}")

    # Try to play a note
    fs.noteon(0, 60, 100)  # channel 0, note 60 (middle C), velocity 100
    print("Successfully played a note")

    # Clean up
    fs.delete()
    print("Successfully cleaned up FluidSynth")

except ImportError as e:
    print(f"Error importing FluidSynth: {e}")
except Exception as e:
    print(f"Error testing FluidSynth: {e}")
